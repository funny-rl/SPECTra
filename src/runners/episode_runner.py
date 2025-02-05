from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from datetime import datetime
import os
import seaborn as sns
import pandas as pd 
import torch as th

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
        
    def get_obs_info(self, observations, actions):
        _obs_dicts = []
        n_agents = observations.shape[1] // 2

        for obs, action in zip(observations, actions):
            own_obs_dict = {}
            own_obs = obs[0]

            if th.all(own_obs == 0):
                own_obs_dict["survive"] = False
                own_obs_dict["action"] = 0
                _obs_dicts.append(own_obs_dict)
                continue
            else:
                own_obs_dict["survive"] = True  
                own_obs_dict["action"] = action

            own_obs_dict["health"] = own_obs[0]
            own_obs_dict["own_location"] = own_obs[1:3]
            unit_type = np.where(own_obs[3:6] == 1)[0][0]
            _sight_range = sight_range(unit_type)
            own_obs_dict["attack_range"] = shoot_range(unit_type) / _sight_range
            own_obs_dict["type"] = get_unit_type(unit_type)

            ally_obs_dicts = []
            for ally_obs in obs[1: n_agents + 1]: 
                ally_obs_dict = {}
                if th.all(ally_obs == 0):
                    ally_obs_dict["not_observed"] = True
                    ally_obs_dicts.append(ally_obs_dict)
                    continue

                ally_obs_dict["relative_location"] = ally_obs[2:4]
                ally_obs_dict["health"] = ally_obs[4]
                ally_type = np.where(ally_obs[5:] == 1)[0][0]
                ally_obs_dict["type"] = get_unit_type(ally_type)
                ally_obs_dicts.append(ally_obs_dict)

            enemy_obs_dicts = []
            for enemy_obs in obs[n_agents + 1:]:
                enemy_obs_dict = {}
                
                if th.all(enemy_obs == 0):
                    enemy_obs_dict["not_observed"] = True
                    enemy_obs_dicts.append(enemy_obs_dict)
                    continue

                enemy_obs_dict["relative_location"] = enemy_obs[2:4]
                enemy_obs_dict["health"] = enemy_obs[4]
                enemy_type = np.where(enemy_obs[5:] == 1)[0][0]
                enemy_obs_dict["type"] = get_unit_type(enemy_type)
                enemy_obs_dicts.append(enemy_obs_dict)

            own_obs_dict["ally"] = ally_obs_dicts  
            own_obs_dict["enemy"] = enemy_obs_dicts

            _obs_dicts.append(own_obs_dict)

        return _obs_dicts
                
            
                
    def run(self, test_mode=False, sub_mac = None, id = None):
        self.reset()

        terminated = False
        episode_return = 0
        
        self.mac.init_hidden(batch_size=self.batch_size, n_agents = self.n_agents)
        if sub_mac is not None:
            sub_mac.init_hidden(batch_size=self.batch_size, n_agents = self.n_agents)
            
            self.mac.load_models(self.mac.agent.args.load_dir)
            sub_mac.load_models(sub_mac.agent.args.load_dir)
            
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
                num_agents = actions.shape[1]
                
                main_observation = self.mac.input_record()
                main_attention_score = self.mac.extract_attention_score()
                main_attention_score = main_attention_score.reshape(num_agents, self.mac.agent.n_head, 1 , -1) # torch.Size([head, num_agents, 1, num_entities]) 
                main_obs_dict = self.get_obs_info(main_observation, actions[0])
                
                sub_actions = sub_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, id = id)
                sub_observation = sub_mac.input_record()
                sub_obs_dict = self.get_obs_info(sub_observation, sub_actions[0])    
                sub_attention_score = sub_mac.extract_attention_score()  
                map_size = sub_attention_score.shape[-1]
                sub_attention_score = sub_attention_score.reshape(-1, sub_mac.agent.args.transformer_heads, map_size, map_size)[:, :, :map_size -1, :map_size - 1]    
                
                self.save_attention_maps(
                    main_attention_score, 
                    os.path.join(save_path, 
                    f"step_{self.t}_SS-VDN"), 
                    "SS-VDN", 
                    self.t,
                    main_obs_dict
                )
                self.save_attention_maps(
                    sub_attention_score, 
                    os.path.join(save_path, 
                    f"step_{self.t}_UPDeT-VDN"), 
                    f"UPDeT-VDN", 
                    self.t,
                    sub_obs_dict
                )        

            cpu_actions = actions.to("cpu").numpy()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

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


    def save_attention_maps(self, attention_scores, base_dir, model_name, step, obs_dict):
        """
        Save attention maps as images using heatmap visualization and export data to CSV files per step.

        :param attention_scores: Tensor of shape [num_agents, num_heads, 1, num_entities]
        :param base_dir: Base directory where step folders will be created
        :param model_name: Name of the model (e.g., "mac" or "submac")
        :param step: Current training step to organize files in separate folders
        """
        # Create directory for the specific step
        os.makedirs(base_dir, exist_ok=True)

        num_agents, num_heads , _, _ = attention_scores.shape

        for agent_idx in range(num_agents):
            
            if not obs_dict[agent_idx].get("survive", False):
                return 

            agent_dir = os.path.join(base_dir, f"agent_{agent_idx}")
            os.makedirs(agent_dir, exist_ok=True)
            
            # Create a large plot with subplots for each head
            fig, axs = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
            if num_heads == 1:  # Ensure axs is iterable
                axs = [axs]

            # Prepare a DataFrame to store all attention scores for this agent
            combined_data = []

            for head_idx in range(num_heads):
                attention_map = attention_scores[agent_idx, head_idx].detach().cpu().numpy().T

                print(f"Step {step} | Agent {agent_idx} | Head {head_idx} | Attention Map Shape: {attention_map.shape}")

                # Plot each attention map in a subplot
                sns.heatmap(attention_map, cmap="Reds", cbar_kws={'label': 'Attention Intensity'}, 
                            xticklabels=False, yticklabels=False, square=True, linewidths=0.5, linecolor='black', ax=axs[head_idx])
                axs[head_idx].set_title(f"{model_name} - Agent {agent_idx} - Head {head_idx}")

                # Append data for CSV
                combined_data.append(attention_map.flatten())

            # Layout adjustment to avoid overlap and ensure the plot fits well
            plt.tight_layout()

            # Save the combined plot
            plot_file = f"agent_{agent_idx}_attention_maps.png"
            plt.savefig(os.path.join(agent_dir, plot_file), bbox_inches='tight')
            plt.close()

            # Save the combined attention scores as a CSV file
            df = pd.DataFrame(np.array(combined_data).T, columns=[f"Head_{i}" for i in range(num_heads)])
            csv_file = f"agent_{agent_idx}_attention_scores.csv"
            df.to_csv(os.path.join(agent_dir, csv_file), index=False)
            
            save_observation_info(obs_dict[agent_idx], agent_dir, agent_idx)



def save_observation_info(obs_dict, base_dir, agent_idx):
    if not obs_dict.get("survive", True):
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    own_x, own_y = obs_dict["own_location"]
    ax.scatter(own_x, own_y, color='blue', label=f"Agent {agent_idx} ({obs_dict['type']})", s=100)
    ax.annotate(f"Agent {agent_idx}\n{obs_dict['type']}\nHP:{obs_dict['health']:.1f}", 
                (own_x, own_y), textcoords="offset points", xytext=(3, 3), ha='right',
                fontsize=8,  
                bbox=dict(facecolor='white', alpha=0.7, pad=0.3))
    
    attack_range = obs_dict["attack_range"]
    action = obs_dict["action"]
    
    attack_circle = patches.Circle((own_x, own_y), attack_range, fill=False, color='blue', linestyle='--', alpha=0.5)
    ax.add_patch(attack_circle)

    x_positions = [own_x]
    y_positions = [own_y]

    for i, ally in enumerate(obs_dict.get("ally", [])):
        if ally.get("not_observed", False):
            continue
        ally_id = i + 1 if i >= agent_idx else i

        ally_x, ally_y = own_x + ally["relative_location"][0], own_y + ally["relative_location"][1]
        ax.scatter(ally_x, ally_y, color='green', marker='^', s=80)
        ax.annotate(f"Ally {ally_id}\n{ally['type']}\nHP:{ally['health']:.1f}",
                    (ally_x, ally_y), textcoords="offset points", xytext=(5, 5), ha='right',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=0.3))
        x_positions.append(ally_x)
        y_positions.append(ally_y)

    for j, enemy in enumerate(obs_dict.get("enemy", [])):
        if enemy.get("not_observed", False):
            continue
        enemy_x, enemy_y = own_x + enemy["relative_location"][0], own_y + enemy["relative_location"][1]
        ax.scatter(enemy_x, enemy_y, color='red', marker='x', s=80)
        ax.annotate(f"Enemy {j}\n{enemy['type']}\nHP:{enemy['health']:.1f}",
                    (enemy_x, enemy_y), textcoords="offset points", xytext=(5, 5), ha='right',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=0.3))
        x_positions.append(enemy_x)
        y_positions.append(enemy_y)

    ax.margins(0.2)
    ax.set_xlim(min(x_positions) - 0.01, max(x_positions) + 0.01)
    ax.set_ylim(min(y_positions) - 0.01, max(y_positions) + 0.01)

    handles = [
        plt.Line2D([], [], marker='o', color='w', label='Agent', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='^', color='w', label='Ally', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='x', color='red', label='Enemy', markersize=10),
        plt.Line2D([], [], color='black', label=f"Action: {action}", linestyle='') 
    ]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0.)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    plt.savefig(os.path.join(base_dir, f"obs_agent_{agent_idx}_observation.png"), bbox_inches='tight')
    plt.close()

    data = []
    
    own_x, own_y = obs_dict["own_location"]
    action = obs_dict["action"]
    attack_range = obs_dict["attack_range"]
    
    # Agent 데이터 추가
    data.append({
        "Agent Index": agent_idx,
        "Type": obs_dict["type"],
        "Health": obs_dict["health"],
        "X Position": own_x,
        "Y Position": own_y,
        "Attack Range": attack_range,
        "Action": action
    })

    for i, ally in enumerate(obs_dict.get("ally", [])):
        if ally.get("not_observed", False):
            continue
        ally_x, ally_y = own_x + ally["relative_location"][0], own_y + ally["relative_location"][1]
        ally_data = {
            "Agent Index": agent_idx,
            "Ally Index": i + 1 if i >= agent_idx else i,
            "Type": ally["type"],
            "Health": ally["health"],
            "X Position": ally_x,
            "Y Position": ally_y,
            "Attack Range": attack_range,
            "Action": action
        }
        data.append(ally_data)

    for j, enemy in enumerate(obs_dict.get("enemy", [])):
        if enemy.get("not_observed", False):
            continue
        enemy_x, enemy_y = own_x + enemy["relative_location"][0], own_y + enemy["relative_location"][1]
        enemy_data = {
            "Agent Index": agent_idx,
            "Enemy Index": j,
            "Type": enemy["type"],
            "Health": enemy["health"],
            "X Position": enemy_x,
            "Y Position": enemy_y,
            "Attack Range": attack_range,
            "Action": action
        }
        data.append(enemy_data)

    file_path = os.path.join(base_dir, f"obs_agent_{agent_idx}_observation.csv")
    header = [
        "Agent Index", "Ally Index", "Enemy Index", "Type", "Health", 
        "X Position", "Y Position", "Attack Range", "Action"
    ]

    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)

def get_unit_type(type):
    if type == 0:
        return "marine" 
    elif type == 1:
        return "marauder"
    elif type == 2:
        return "medivac"

def sight_range(type):
    if type == 0:
        return 9
    elif type == 1:
        return 9
    elif type == 2:
        return 9
    
def shoot_range(type):
    if type == 0:
        return 6
    elif type == 1:
        return 6
    elif type == 2:
        return 6