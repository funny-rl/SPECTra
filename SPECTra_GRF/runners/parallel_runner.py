from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import wandb 
import json

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger, scenario_manager = None):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.scenario_manager = scenario_manager
        self.from_scratch = False

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.test_win_rate = []
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode):
        self.batch = self.new_batch()

        self.t = 0
        self.env_steps_this_run = 0

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        check_epi_loc = []
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            check_epi_loc.append(data["state"][:, -15:-13])
            
        self.batch.update(pre_transition_data, ts=0)
                
    def run(self, test_mode=False):
        self.reset(test_mode)
        
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size, n_agents = self.args.n_agents)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        win_rate = []
        (
            total_own_changing_r,
            total_oob_r,
            total_pass_r,
            total_player_pos_r,
            total_yellow_r,
            total_ball_position_r,
            total_score_r,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", (cpu_actions[action_idx], action_idx)))
                    action_idx += 1
            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            
            if all_terminated:
                break

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
            
            (
                batch_own_changing_r,
                batch_oob_r,
                batch_pass_r,
                batch_player_pos_r,
                batch_yellow_r,
                batch_ball_position_r,
                batch_score_r,
            ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    own_changing_r, oob_r, pass_r, player_pos_r, yellow_r, ball_position_r, score_r = data["reward"]

                    batch_own_changing_r += own_changing_r
                    batch_oob_r += oob_r
                    batch_pass_r += pass_r
                    batch_player_pos_r += player_pos_r
                    batch_yellow_r += yellow_r
                    batch_ball_position_r += ball_position_r
                    batch_score_r += score_r
                    
                    if data["terminated"]:
                        if score_r > 0:
                            win_rate.append(1)
                        else:
                            win_rate.append(0)
                    if self.args.n_reward == 1:
                        data_reward = sum(data["reward"])
                        episode_returns[idx] += data_reward
                    else:
                        data_reward = data["reward"]
                        episode_returns[idx] += sum(data_reward)
                    post_transition_data["reward"].append((data_reward,))
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
            
            # analyze earch reward
            total_own_changing_r += batch_own_changing_r
            total_oob_r += batch_oob_r 
            total_pass_r += batch_pass_r
            total_player_pos_r += batch_player_pos_r
            total_yellow_r += batch_yellow_r 
            total_ball_position_r += batch_ball_position_r
            total_score_r += batch_score_r
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run
        else:
            self.test_win_rate.append(win_rate)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        infos = [{k: v for k, v in d.items() if k == 'score_reward'} for d in infos]
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            self.logger.log_stat("test_win_rate", np.mean(self.test_win_rate), self.t_env)
            self.test_win_rate = []
            
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:

            self._log(cur_returns, cur_stats, log_prefix)
                
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.logger.log_stat("ownership_changing_reward", total_own_changing_r, self.t_env)
                self.logger.log_stat("out_of_bound_reward", total_oob_r, self.t_env)
                self.logger.log_stat("pass_reward", total_pass_r, self.t_env)
                self.logger.log_stat("player_position_reward", total_player_pos_r, self.t_env)
                self.logger.log_stat("yellow_card_reward", total_yellow_r, self.t_env)
                self.logger.log_stat("ball_position_reward", total_ball_position_r, self.t_env)
                self.logger.log_stat("score_reward", total_score_r, self.t_env)
                self.logger.log_stat("train_win_rate", np.mean(win_rate), self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
        
    def supervised_learning(self):
        import os
        import torch.nn as nn
        import torch.optim as optim
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        # Set device to GPU if available
        device = self.args.device

        data_path = os.path.join(os.getcwd(), "data/real_data")
        data_tensor = torch.tensor(
            pd.read_csv(os.path.join(data_path, "dataset_data.csv")).values, dtype=torch.float32
        ).to(
            device
        )  # Move to GPU
        label_tensor = torch.tensor(
            pd.read_csv(os.path.join(data_path, "dataset_label.csv")).values, dtype=torch.int8
        ).to(
            device
        )  # Move to GPU

        test_data_tensor = torch.tensor(
            pd.read_csv(os.path.join(data_path, "test_dataset_data.csv")).values, dtype=torch.float32
        ).to(
            device
        )  # Move to GPU
        test_label_tensor = torch.tensor(
            pd.read_csv(os.path.join(data_path, "test_dataset_label.csv")).values, dtype=torch.int8
        ).to(
            device
        )  # Move to GPU
        train_dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
        test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_label_tensor)

        num_epochs = 100
        eval_interval = 1
        batch_size = 5120
        patience = 5
        best_accuracy = 0.0
        patience_counter = 0

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.mac.agent.parameters(), lr=self.args.lr)

        self.mac.agent.to(device)

        self.mac.set_train_mode()

        for epoch in range(num_epochs):
            total_loss = 0.0

            total_correct = 0
            total_samples = 0

            self.mac.hidden_states = None
            self.mac.init_hidden(batch_size=int(batch_size / self.args.n_agents), n_agents=self.args.n_agents)

            for batch in train_dataloader:
                optimizer.zero_grad()

                x, y = batch
                if x.shape[0] != batch_size:
                    continue
                outputs = self.mac.supervised_select_actions(x).reshape(-1, 19)

                one_hot_label = F.one_hot(y.to(torch.long), num_classes=19).reshape(
                    int(batch_size / self.args.n_agents), self.args.n_agents, 19
                )

                self.mac.hidden_states = self.mac.hidden_states.detach()

                total_samples += batch_size

                indices = torch.argmax(outputs, dim=-1)

                total_correct += (indices == y.reshape(-1)).sum().item()

                loss = criterion(outputs, y.to(torch.long).reshape(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)

            accuracy = total_correct / total_samples
            self.logger.log_stat("Imitation Train accuracy", accuracy, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {accuracy:.4f}")

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            self.logger.log_stat("Imitation Train Loss", avg_loss, epoch)

            if (epoch + 1) % eval_interval == 0:
                self.mac.set_evaluation_mode()
                total_correct = 0
                total_samples = 0

                self.mac.hidden_states = None
                self.mac.init_hidden(batch_size=int(batch_size / self.args.n_agents), n_agents=self.args.n_agents)

                action_counts = torch.zeros(19, device=device)

                with torch.no_grad():
                    for test_batch in test_dataloader:
                        x_test, y_test = test_batch
                        if x_test.shape[0] != batch_size:
                            continue

                        outputs_test = self.mac.supervised_select_actions(x_test).reshape(-1, 19)

                        indices = torch.argmax(outputs_test, dim=-1)
                        flatten_indices = indices.reshape(-1)
                        action_counts += torch.bincount(flatten_indices, minlength=19)

                        total_samples += batch_size
                        total_correct += (indices == y_test.reshape(-1)).sum().item()

                action_counts_np = action_counts.cpu().numpy()
                for action_index, count in enumerate(action_counts_np):
                    if action_index < 9:
                        self.logger.log_stat(f"Action {action_index} Count", count, epoch)

                accuracy = total_correct / total_samples
                self.logger.log_stat("Imitation Eval accuracy", accuracy, epoch)
                print(f"Epoch [{epoch+1}/{num_epochs}], Eval Accuracy: {accuracy:.4f}")

                # Early stopping condition
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    # for param_group in optimizer.param_groups:
                    #     param_group["lr"] *= 0.9
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}, best accuracy: {best_accuracy:.4f}")
                        del data_tensor, label_tensor, test_data_tensor, test_label_tensor
                        self.mac.agent.to("cpu")
                        torch.cuda.empty_cache()
                        return

        del data_tensor, label_tensor, test_data_tensor, test_label_tensor
        self.mac.agent.to("cpu")
        torch.cuda.empty_cache()
        return

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions, batch_idx = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions, batch_idx)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info,
                "avail_actions": avail_actions,
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "obs": env.get_obs(),
                "avail_actions": env.get_avail_actions()
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

