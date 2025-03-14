import datetime
import os
import glob
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, ReplayBuffer_Prior
from components.transforms import OneHot

from run.generate_scenario import Scenario_Manager

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = args.GPU if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    args.log_dir = f"{args.env_args['env_name']}/{args.name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    args.unique_token = str(int(time.time()))

    if args.use_wandb:
        wandb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, args.log_dir)
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


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    
    scenario_manager = Scenario_Manager(args = args)
    args.env_args["env_name"] = scenario_manager.new_env_name 
    args.env_args["batch_size_run"] = args.batch_size_run
        
    if args.env_args["logdir"] is not None:
        args.env_args["logdir"] = os.path.join(args.env_args["logdir"], args.env, args.name, args.group_name, args.unique_token)
        os.makedirs(args.env_args["logdir"], exist_ok=True)

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger, scenario_manager = scenario_manager)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    args.obs_component = env_info["obs_component"]
    args.state_component = env_info["state_component"]

    args.n_agents = env_info["n_agents"]
    args.n_allies = env_info["n_allies"]
    args.n_enemies = env_info["n_enemies"]
    
    args.n_actions = env_info["n_actions"]
    args.episode_limit = env_info["episode_limit"]
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (args.n_reward,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    if 'prior' in args.name:
        buffer_prior = ReplayBuffer_Prior(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                        args.burn_in_period,
                                        preprocess=preprocess,
                                        device="cpu" if args.buffer_cpu_only else args.device,
                                        alpha=args.alpha)

        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            args.burn_in_period,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device)

    else:
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            args.burn_in_period,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        if 'prior' in args.name:
            buffer.insert_episode_batch(episode_batch)
            buffer_prior.insert_episode_batch(episode_batch)
        else:
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):

            if 'prior' in args.name:
                idx, episode_sample = buffer_prior.sample(args.batch_size)
            else:
                episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            if 'prior' in args.name:
                update_prior = learner.train(
                    episode_sample, runner.t_env, episode)
                buffer_prior.update_priority(
                    idx, update_prior.to('cpu').tolist())
            else:
                learner.train(episode_sample, runner.t_env, episode)


        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.name, args.group_name, args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            if args.double_q:
                os.makedirs(save_path + '_x', exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run 

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available(): 
        config["use_cuda"] = False
        _log.info(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]
    return config
