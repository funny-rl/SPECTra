from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process

import numpy as np
import time


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger, eval_args = None):
        self.args = args
        self.eval_args = eval_args
        
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker,
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)
        if self.args.evaluate:
            print("Waiting the environment to start...")
            time.sleep(5)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.n_agents = self.env_info["n_agents"]
        
        if self.eval_args is not None:
            self.eval_parent_conns, self.eval_worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
            eval_env_fn = env_REGISTRY[self.eval_args.env]
            self.eval_ps = []
            for i, eval_worker_conn in enumerate(self.eval_worker_conns):
                ps = Process(target=env_worker,
                             args=(eval_worker_conn, CloudpickleWrapper(partial(eval_env_fn, **self.eval_args.env_args))))
                self.eval_ps.append(ps)
            for p in self.eval_ps:
                p.daemon = True
                p.start()
            
            self.eval_parent_conns[0].send(("get_env_info", None))
            self.eval_env_info = self.eval_parent_conns[0].recv()
            self.eval_episode_limit = self.eval_env_info["episode_limit"]
            self.eval_n_agents = self.eval_env_info["n_agents"]
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        if self.args.use_cuda and not self.args.cpu_inference:
            self.batch_device = self.args.device
        else:
            self.batch_device = "cpu" if self.args.buffer_cpu_only else self.args.device
        print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& self.batch_device={}".format(
            self.batch_device))
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.batch_device)
        self.mac = mac
        
    def eval_setup(self, scheme, groups, preprocess, mac):
        if self.args.use_cuda and not self.args.cpu_inference:
            self.batch_device = self.args.device
        else:
            self.batch_device = "cpu" if self.args.buffer_cpu_only else self.args.device
        print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& self.batch_device={}".format(
            self.batch_device))
        self.eval_new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.batch_device)
        self.mac = mac

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode):
        if not test_mode or not self.args.use_CL:
            self.batch = self.new_batch()
        else:
            self.batch = self.eval_new_batch()
            
        if (self.args.use_cuda and self.args.cpu_inference) and str(self.mac.get_device()) != "cpu":
            self.mac.cpu()  # copy model to cpu
            
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Reset the envs
        if not test_mode or not self.args.use_CL:
            for parent_conn in self.parent_conns:
                parent_conn.send(("reset", None))
                
            for parent_conn in self.parent_conns:
                data = parent_conn.recv()
                pre_transition_data["state"].append(data["state"])
                pre_transition_data["obs"].append(data["obs"])
                pre_transition_data["avail_actions"].append(data["avail_actions"])
        else:
            for parent_conn in self.eval_parent_conns:
                parent_conn.send(("reset", None))
                
            for parent_conn in self.eval_parent_conns:
                data = parent_conn.recv()
                pre_transition_data["state"].append(data["state"])
                pre_transition_data["obs"].append(data["obs"])
                pre_transition_data["avail_actions"].append(data["avail_actions"])
                
        self.batch.update(pre_transition_data, ts=0, mark_filled=True)

        self.t = 0
        self.env_steps_this_run = 0
        

    def run(self, test_mode=False):
        self.reset(test_mode)
        

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        if not test_mode or not self.args.use_CL:
            self.mac.init_hidden(batch_size=self.batch_size, n_agents = self.n_agents)
        else:
            self.mac.init_hidden(batch_size=self.batch_size, n_agents = self.eval_n_agents)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  
        
        

        save_probs = getattr(self.args, "save_probs", False)
        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                                  test_mode=test_mode)

            cpu_actions = actions.to("cpu").numpy()
            
            # Update the actions taken
            actions_chosen = {
                "actions": np.expand_dims(cpu_actions, axis=1),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            
            # Send actions to each env
            action_idx = 0
            if not test_mode or not self.args.use_CL:
                for idx, parent_conn in enumerate(self.parent_conns):
                    if idx in envs_not_terminated:  # We produced actions for this env
                        if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                            parent_conn.send(("step", cpu_actions[action_idx]))
                        action_idx += 1  # actions is not a list over every env
            else:
                for idx, parent_conn in enumerate(self.eval_parent_conns):
                    if idx in envs_not_terminated:
                        if not terminated[idx]:
                            parent_conn.send(("step", cpu_actions[action_idx]))
                        action_idx += 1
                    
            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            # Receive data back for each unterminated env
            if not test_mode or not self.args.use_CL:
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        data = parent_conn.recv()
                        # Remaining data for this current timestep
                        post_transition_data["reward"].append((data["reward"],))

                        episode_returns[idx] += data["reward"]
                        episode_lengths[idx] += 1
                        if not test_mode:
                            self.env_steps_this_run += 1

                        env_terminated = False
                        if data["terminated"]:
                            final_env_infos.append(data["info"])
                        if data["terminated"] and not data["info"].get("episode_limit", False):
                            env_terminated = True
                        terminated[idx] = data["terminated"]
                        post_transition_data["terminated"].append((env_terminated,))

                        # Data for the next timestep needed to select an action
                        pre_transition_data["state"].append(data["state"])
                        pre_transition_data["avail_actions"].append(data["avail_actions"])
                        pre_transition_data["obs"].append(data["obs"])
            else:
                for idx, parent_conn in enumerate(self.eval_parent_conns):
                    if not terminated[idx]:
                        data = parent_conn.recv()
                        post_transition_data["reward"].append((data["reward"],))

                        episode_returns[idx] += data["reward"]
                        episode_lengths[idx] += 1
                        if not test_mode:
                            self.env_steps_this_run += 1

                        env_terminated = False
                        if data["terminated"]:
                            final_env_infos.append(data["info"])
                        if data["terminated"] and not data["info"].get("episode_limit", False):
                            env_terminated = True
                        terminated[idx] = data["terminated"]
                        post_transition_data["terminated"].append((env_terminated,))

                        pre_transition_data["state"].append(data["state"])
                        pre_transition_data["avail_actions"].append(data["avail_actions"])
                        pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            if self.args.evaluate:
                assert self.batch_size == 1
                move = [["北", "南", "东", "西"][action - 2] if action > 1 and action < 6 else "action-{}".format(action)
                        for action in cpu_actions[0]]
                print(self.t, move, post_transition_data["reward"])
                time.sleep(1)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        if not test_mode or not self.args.use_CL:
            for parent_conn in self.parent_conns:
                parent_conn.send(("get_stats", None))

            env_stats = []
            for parent_conn in self.parent_conns:
                env_stat = parent_conn.recv()
                env_stats.append(env_stat)
        else:
            for parent_conn in self.eval_parent_conns:
                parent_conn.send(("get_stats", None))

            env_stats = []
            for parent_conn in self.eval_parent_conns:
                env_stat = parent_conn.recv()
                env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
            
        return self.batch
        # return clear_no_reward_sub_trajectory(self.batch)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_min", np.min(returns), self.t_env)
        self.logger.log_stat(prefix + "return_max", np.max(returns), self.t_env)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
