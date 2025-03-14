import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import sys

from copy import deepcopy

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from run.curriculum import Curriculum_Manager


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)

    th.set_num_threads(args.thread_num)
    # th.set_num_interop_threads(8)

    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    if args.use_wandb:
        wandb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, unique_token)
        logger.setup_wandb(log_dir = wandb_logs_direc, args = args)

    # sacred is on by default
    logger.setup_sacred(_run)
    
    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)
    
def split_args(args):
    main_args = args.main_model
    sub_args = args.sub_model
    del args.main_model
    del args.sub_model
    
    args1 = deepcopy(args)
    args2 = deepcopy(args)   
    for key, value in main_args.items():
        setattr(args1, key, value)
    for key, value in sub_args.items():
        setattr(args2, key, value)

    return args1, args2

def dual_evaluate_sequential(args, runner, sub_mac):
    for id in range(args.test_nepisode):
        runner.run(test_mode=True, sub_mac = sub_mac, id = id)
        if args.save_replay:
            runner.save_replay()
    runner.close_env()


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)
    if args.save_replay:
        runner.save_replay()
    runner.close_env()
    

def init_env(args, logger, init = False, eval_args = None, runner = None, learner = None):
    use_CL = args.use_CL
    
    if not init and use_CL:
        _t_env = runner.t_env
        _agent = learner.mac
        _target_agent = learner.target_mac
        _mixer = learner.mixer.state_dict()
        _target_mixer = learner.target_mixer.state_dict()
    
    runner = r_REGISTRY[args.runner](args=args, logger=logger, eval_args=eval_args)
    
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    
    if args.env in ["sc2", "sc2_v2", "gfootball"]:
        if args.env in ["sc2", "sc2_v2"]:
            args.output_normal_actions = env_info["n_normal_actions"]
        args.n_enemies = env_info["n_enemies"]
        args.n_allies = env_info["n_allies"]
        # args.obs_ally_feats_size = env_info["obs_ally_feats_size"]
        # args.obs_enemy_feats_size = env_info["obs_enemy_feats_size"]
        args.state_ally_feats_size = env_info["state_ally_feats_size"]
        args.state_enemy_feats_size = env_info["state_enemy_feats_size"]
        args.obs_component = env_info["obs_component"]
        args.state_component = env_info["state_component"]
        args.map_type = env_info["map_type"]
        args.agent_own_state_size = env_info["state_ally_feats_size"]
        
        

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
        
    if use_CL:
        eval_env_info = runner.eval_env_info

        eval_args.n_enemies = eval_env_info["n_enemies"]
        eval_args.n_allies = eval_env_info["n_allies"]
        eval_args.state_ally_feats_size = eval_env_info["state_ally_feats_size"]
        eval_args.state_enemy_feats_size = eval_env_info["state_enemy_feats_size"]
        eval_args.obs_component = eval_env_info["obs_component"]
        eval_args.state_component = eval_env_info["state_component"]
        eval_args.map_type = eval_env_info["map_type"]
        eval_args.agent_own_state_size = eval_env_info["state_ally_feats_size"]        

        eval_scheme = {
            "state": {"vshape": eval_env_info["state_shape"]},
            "obs": {"vshape": eval_env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (eval_env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "probs": {"vshape": (eval_env_info["n_actions"],), "group": "agents", "dtype": th.float},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        eval_groups = {
            "agents": eval_env_info["n_agents"]
        }
        eval_preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=eval_env_info["n_actions"])])
        }
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, eval_args)
        
        if not init:
            runner.t_env = _t_env
            
        runner.eval_setup(scheme=eval_scheme, groups=eval_groups, preprocess=eval_preprocess, mac = mac)
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if not init and use_CL:
        learner.mac.load_state(_agent)
        learner.target_mac.load_state(_target_agent)
        learner.mixer.load_state_dict(_mixer)
        learner.target_mixer.load_state_dict(_target_mixer)
        print(f"Curriculum Upgrade Complete!!!")

    if args.use_cuda:
        learner.cuda()

    if args.transfer_learning and args.load_dir != "":

        model_path = os.path.join(args.load_dir)

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

    return args, runner, buffer, learner


def run_sequential(args, logger):
    use_CL = args.use_CL
    eval_args = None
    if use_CL:
        CL_manager = Curriculum_Manager(args)
        basic_args = args
        args = CL_manager.init_train_args(basic_args)
        eval_args = CL_manager.init_eval_args(basic_args)
    # Init runner so we can get env info
    
    if not args.check_model_attention_map:
        args, runner, buffer,learner  = init_env(args, logger, init = True, eval_args = eval_args)
    else:
        args1, args2 = split_args(args)
        
        args1, runner1, buffer, _  = init_env(args1, logger, init = True, eval_args = eval_args)
        args2, runner2, buffer, _  = init_env(args2, logger, init = True, eval_args = eval_args)
        
        dual_evaluate_sequential(args1, runner1, sub_mac = runner2.mac)
        return 

    if args.evaluate or args.save_replay:
        evaluate_sequential(args, runner)
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
              
    while runner.t_env <= args.t_max:
        if use_CL:
            args, upgrade = CL_manager.update(args, runner.t_env)
            if upgrade:
                args, runner, buffer,learner = init_env(args, logger, init = False, eval_args = eval_args, \
                    runner = runner, learner = learner)
            print(f"Current Curriculum Level: {args.env_args['capability_config']['n_units']}v{args.env_args['capability_config']['n_enemies']}")
                
        # Run for a whole episode at a time
        with th.no_grad():
            # t_start = time.time()
            episode_batch = runner.run(test_mode=False)
            if episode_batch.batch_size > 0:  # After clearing the batch data, the batch may be empty.
                buffer.insert_episode_batch(episode_batch)
            # print("Sample new batch cost {} seconds.".format(time.time() - t_start))
            episode += args.batch_size_run

        if buffer.can_sample(args.batch_size):
            if args.accumulated_episodes and episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            with th.no_grad():
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)
                    th.cuda.empty_cache()

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
            model_save_time = runner.t_env
            local_results_path = os.path.expanduser(args.local_results_path)
            save_path = os.path.join(local_results_path, "models", f"{args.env_args['map_name']}_{args.env_args['capability_config']['n_units']}_{args.env_args['capability_config']['n_enemies']}",args.name,str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            
            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        th.cuda.empty_cache()
            
    runner.close_env()
    logger.console_logger.info("Finished Training")

    # flush
    sys.stdout.flush()
    time.sleep(10)


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]
        
    if config["name"] == "attention_map":
        config["check_model_attention_map"] = True
    
    if config["check_model_attention_map"]:
        config["evaluate"] = True
        config["use_CL"] = False
        config["save_replay"] = True
        config["runner"] = "episode"
        config["batch_size_run"] = 1 
        config["batch_size"] = 1
        
    return config
