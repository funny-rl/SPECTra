from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import os
import seaborn as sns

class EpisodeRunner:

    def __init__(self, args, logger, eval_args = None):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        if self.batch_size > 1:
            self.batch_size = 1
            logger.console_logger.warning("Reset the `batch_size_run' to 1...")

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if self.args.evaluate:
            print("Waiting the environment to start...")
            time.sleep(5)
        self.episode_limit = self.env.episode_limit
        
        self.env_info = self.get_env_info()
        self.n_agents = self.env_info["n_agents"]        

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

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
        
    def test_setup(self, scheme, groups, preprocess, mac):
        
        if self.args.use_cuda and not self.args.cpu_inference:
            self.batch_device = self.args.device
        else:
            self.batch_device = "cpu" if self.args.buffer_cpu_only else self.args.device
            
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.batch_device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        if (self.args.use_cuda and self.args.cpu_inference) and str(self.mac.get_device()) != "cpu":
            self.mac.cpu()  # copy model to cpu

        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, sub_mac = None, id = None):
        self.reset()

        terminated = False
        episode_return = 0
        
        self.mac.init_hidden(batch_size=self.batch_size, n_agents = self.n_agents)
        if sub_mac is not None:
            sub_mac.init_hidden(batch_size=self.batch_size, n_agents = self.n_agents)
            
            now = datetime.now()
            map_name = self.args.env_args["map_name"]
            time_string = now.strftime("%Y-%m-%d %H:%M:%S")
            local_results_path = os.path.expanduser(self.args.local_results_path)
            save_path = os.path.join(local_results_path, "attention_score", f"{map_name}_{self.args.env_args['capability_config']['n_units']}", time_string)
            os.makedirs(save_path, exist_ok=True)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, id = id)
            if sub_mac is not None:

                _ = sub_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, id = id)

                attention_score1 = self.mac.extract_attention_score().reshape(4, -1, 1 , 10) # torch.Size([head, num_agents, 1, num_entities]) 
                self.save_attention_maps(attention_score1, os.path.join(save_path, f"step_{self.t}_SS-VDN"), "SS-VDN")
                
                attention_score2 = sub_mac.extract_attention_score()
                for idx, attention_score_block in enumerate(attention_score2):
                    attention_score_block = attention_score_block.reshape(-1, 3, 11, 11).permute(1, 0, 2, 3)  # torch.Size([head, num_agents, 1, num_entities]) 
                    self.save_attention_maps(attention_score_block, os.path.join(save_path, f"step_{self.t}_UPDeT-VDN_{idx}"), f"UPDeT-VDN_{idx}")             
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            if self.args.evaluate:
                time.sleep(1)
                print(self.t, post_transition_data["reward"])

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

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


    def save_attention_maps(self, attention_scores, save_dir, model_name):
        """
        Save attention maps as images using heatmap visualization.
        
        :param attention_scores: Tensor of shape [num_agents, num_heads, 1, num_entities]
        :param save_dir: Directory where images will be saved
        :param model_name: Name of the model (e.g., "mac" or "submac")
        """
        os.makedirs(save_dir, exist_ok=True)
        num_heads, num_agents, _, num_entities = attention_scores.shape
        for head_idx in range(num_heads):
            for agent_idx in range(num_agents):
                # 텐서를 detach하고 NumPy로 변환
                attention_map = attention_scores[head_idx, agent_idx].detach().cpu().numpy().T

                print(f"Head {head_idx}, Agent {agent_idx}, Attention Map Shape: {attention_map.shape}")

                # Set up the plot
                plt.figure(figsize=(10, 10))

                # Use seaborn heatmap for better visualization
                sns.heatmap(attention_map, cmap="Reds", cbar_kws={'label': 'Attention Intensity'}, 
                            xticklabels=False, yticklabels=False, square=True, linewidths=0.5, linecolor='black')

                plt.title(f"{model_name} - Agent {agent_idx} - Head {head_idx}")

                # Layout adjustment to avoid overlap and ensure the plot fits well
                plt.tight_layout()

                # Save the figure
                file_name = f"head_{head_idx}_agent_{agent_idx}.png"
                plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
                plt.close()