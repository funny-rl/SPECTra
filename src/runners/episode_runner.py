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
        
        if observations.shape[-1] == 11:
            race = "protoss"
            own_health_idx = 0
            own_shield_idx = 1
            own_pos_start_idx = 2
            own_pos_end_idx = 4
            own_type_start_idx = 4
            own_type_end_idx = 7
            
            ally_distance_idx = 1
            ally_pos_sta_idx = 2
            ally_pos_end_idx = 4
            ally_health_idx = 4
            ally_shield_idx = 5
            ally_type_sta_idx = 6
            ally_type_end_idx = 9
            
            en_dis_idx = 1
            en_pos_sta_idx = 2
            en_pos_end_idx = 4
            en_health_idx = 4
            en_shield_idx = 5
            en_type_sta_idx = 6
            en_type_end_idx = 9
     
        else:
            raise NotImpelentedError
        
        _obs_dicts = []
        
        assert actions.shape[0] == 1, "batch must be 1."
        actions = actions[0] 
        n_agents = observations.shape[1] // 2
        n_allies = n_agents - 1

        for idx, obs in enumerate(observations):
            own_obs_dict = {}
            own_obs = obs[0] # own information of observation
            action = actions[idx]

            if th.all(own_obs == 0):
                own_obs_dict["survive"] = False
                assert action == 0, f"Dead agent can't do action {action}."
                own_obs_dict["action"] = 0
                _obs_dicts.append(own_obs_dict)
                continue
            else:
                own_obs_dict["survive"] = True  
                assert action != 0, f"Dead agent can't do action 0."
                own_obs_dict["action"] = action

            # Related HP
            own_obs_dict["health"] = own_obs[own_health_idx]           
            assert 0 <= own_obs_dict["health"] <= 1 , f"health percentage can't be {own_obs_dict['health']}%"
            if race == "protoss":
                own_obs_dict["shield"] = own_obs[own_shield_idx]
                assert 0 <= own_obs_dict["shield"] <= 1 , f"shield percentage can't be {own_obs_dict['shield']}%"
            
            
            # Related Location
            own_obs_dict["own_location"] = own_obs[own_pos_start_idx:own_pos_end_idx] 
            assert all(0 <= x <= 1 for x in own_obs_dict["own_location"]), "Agent escapes obs boundary."
            # Type Location
            own_type = own_obs[own_type_start_idx:own_type_end_idx]
            assert own_type.sum() == 1, "Error: There should be exactly one '1' in the slice."
            unit_type = np.where(own_type == 1)[0][0]
            own_obs_dict["type"] = get_unit_name(unit_type, race)
            
            # attack_range
            own_obs_dict["attack_range"] = 0.6 / 0.9 # fixed value because we don't use 'use_unit_ranges' variable

            ally_obs_dicts = []
            for ally_obs in obs[1: n_agents]: 
                ally_obs_dict = {}
                if th.all(ally_obs == 0):
                    ally_obs_dict["not_observed"] = True
                    ally_obs_dicts.append(ally_obs_dict)
                    continue
                #distance
                assert ally_obs[ally_distance_idx] <= 1, "Scaled distance can't be greater than 1."
                # related location
                ally_obs_dict["relative_location"] = ally_obs[ally_pos_sta_idx:ally_pos_end_idx]
                assert all(-1 <= x <= 1 for x in ally_obs_dict["relative_location"]), "Ally escapes obs boundary."
                # related ally health
                ally_obs_dict["health"] = ally_obs[ally_health_idx]
                assert 0 <= ally_obs_dict["health"] <= 1 , f"health percentage can't be {ally_obs_dict['health']}%"
                # related ally shield
                if race == "protoss":
                    ally_obs_dict["shield"] = ally_obs[ally_shield_idx]
                    assert 0 <= ally_obs_dict["shield"] <= 1 , f"shield percentage can't be {ally_obs_dict['shield']}%"
                # related ally type
                ally_type = ally_obs[ally_type_sta_idx:ally_type_end_idx]
                assert ally_type.sum() == 1, "Error: There should be exactly one '1' in the slice."
                ally_id = np.where(ally_type == 1)[0][0]
                ally_obs_dict["type"] = get_unit_name(ally_id, race)
                ally_obs_dicts.append(ally_obs_dict)
                
            assert len(ally_obs_dicts) == n_agents - 1, f"Len of ally_obs_dicts must be {n_allies}, Not {len(ally_obs_dicts)}."

            enemy_obs_dicts = []
            for enemy_obs in obs[n_agents:]:
                enemy_obs_dict = {}
                
                if th.all(enemy_obs == 0):
                    enemy_obs_dict["not_observed"] = True
                    enemy_obs_dicts.append(enemy_obs_dict)
                    continue
                # distance
                assert enemy_obs[en_dis_idx] <= 1, "Scaled distance can't be greater than 1."
                # related location
                enemy_obs_dict["relative_location"] = enemy_obs[en_pos_sta_idx:en_pos_end_idx]
                assert all(-1 <= x <= 1 for x in enemy_obs_dict["relative_location"]), "Enemy escapes obs boundary."
                # related enemy health
                enemy_obs_dict["health"] = enemy_obs[en_health_idx]
                assert 0 <= enemy_obs_dict["health"] <= 1 , f"health percentage can't be {enemy_obs_dict['health']}%"
                # related enemy shield
                if race == "protoss":
                    enemy_obs_dict["shield"] = enemy_obs[en_shield_idx]
                    assert 0 <= enemy_obs_dict["shield"] <= 1 , f"shield percentage can't be {enemy_obs_dict['shield']}%"
                # related enemy type
                enemy_type = enemy_obs[en_type_sta_idx:en_type_end_idx]
                assert enemy_type.sum() == 1, "Error: There should be exactly one '1' in the slice."
                enemy_id = np.where(enemy_type== 1)[0][0]
                enemy_obs_dict["type"] = get_unit_name(enemy_id, race)
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
                
                main_obs = self.mac.get_agent_obs().cpu()
                main_attention_score = self.mac.extract_attention_score().cpu()
                main_attention_score = main_attention_score.reshape(self.mac.agent.n_head, num_agents, main_attention_score.shape[1] , main_attention_score.shape[2]).permute(1, 0, 2, 3).cpu()
                main_obs_dict = self.get_obs_info(main_obs, actions.cpu())
                
                sub_actions = sub_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, id = id)

                sub_obs = sub_mac.get_agent_obs().cpu()
                sub_attention_score = sub_mac.extract_attention_score().cpu()
                sub_attention_score = sub_attention_score.reshape(num_agents, sub_mac.args.transformer_heads, sub_attention_score.shape[1] , sub_attention_score.shape[2]).permute(0, 1, 3, 2)[:, : , :-1, :-1].cpu()
                sub_obs_dict = self.get_obs_info(sub_obs, sub_actions.cpu())

                self.save_game_info(
                    main_attention_score, 
                    os.path.join(save_path, 
                    f"step_{self.t}_SS-VDN"), 
                    "SS-VDN", 
                    self.t,
                    main_obs_dict, 
                    sub_obs_dict
                )
                self.save_game_info(
                    sub_attention_score, 
                    os.path.join(save_path, 
                    f"step_{self.t}_UPDeT-VDN"), 
                    f"UPDeT-VDN", 
                    self.t,
                    sub_obs_dict,
                    main_obs_dict
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


    def save_game_info(self, attention_scores, base_dir, model_name, step, obs_dicts, opponent_obs_dicts):
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

        for agent_idx, obs_dict  in enumerate(obs_dicts):
            
            if not obs_dict.get("survive", False):
                continue
            
            if obs_dict.get("action") == opponent_obs_dicts[agent_idx].get("action"):
                continue

            agent_dir = os.path.join(base_dir, f"agent_{agent_idx}")
            os.makedirs(agent_dir, exist_ok=True)
            
            fig, axs = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
            if num_heads == 1:  # Ensure axs is iterable
                axs = [axs]

            # Prepare a DataFrame to store all attention scores for this agent
            attention_data_set = []

            for head_idx in range(num_heads):
                attention_map = attention_scores[agent_idx, head_idx].detach().cpu().numpy().T

                print(f"Step {step} | Agent {agent_idx} | Head {head_idx}")

                sns.heatmap(attention_map, cmap="Reds", cbar=False, 
                            xticklabels=False, yticklabels=False, annot=True, fmt=".2f",square=True, linewidths=0.5, linecolor='black', ax=axs[head_idx])
                axs[head_idx].set_title(f"{model_name} - Agent {agent_idx} - Head {head_idx}")
                attention_data_set.append(attention_map.flatten())

            fig.colorbar(axs[0].collections[0], cax=fig.add_axes([0.93, 0.15, 0.03, 0.7]), label='Attention scores')
            
            plt.tight_layout()

            plot_file = f"Step_{step}_agent_{agent_idx}_attention_maps.png"
            plt.savefig(os.path.join(agent_dir, plot_file), bbox_inches='tight')
            plt.close(fig)

            # Save the combined attention scores as a CSV file
            df = pd.DataFrame(np.array(attention_data_set).T, columns=[f"Head_{i}" for i in range(num_heads)])
            csv_file = f"agent_{agent_idx}_attention_scores.csv"
            df.to_csv(os.path.join(agent_dir, csv_file), index=False)
            
            save_observation_info(obs_dict, agent_dir, agent_idx)

def save_observation_info(obs_dict, base_dir, agent_idx):

    fig, ax = plt.subplots(figsize=(8, 8))

    own_x, own_y = obs_dict["own_location"]
    ax.scatter(own_x, own_y, color='blue', label=f"Agent {agent_idx} ({obs_dict['type']})", s=100)
    ax.annotate(f"Agent {agent_idx}\n{obs_dict['type']}\nHP:{obs_dict['health']:.1f}", 
                (own_x, own_y), textcoords="offset points", xytext=(0, 5), ha='center',
                fontsize=8,  
                bbox=dict(facecolor='white', alpha=1.0, pad=0.3))
    
    attack_range = obs_dict["attack_range"]
    action = obs_dict["action"] 
    
    attack_circle = patches.Circle((own_x, own_y), attack_range, fill=False, color='blue', linestyle='--', alpha=0.2)
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
                    (ally_x, ally_y), textcoords="offset points", xytext=(0, 5), ha='center',
                    fontsize=8, bbox=dict(facecolor='white', alpha=1.0, pad=0.3))
        x_positions.append(ally_x)
        y_positions.append(ally_y)

    for j, enemy in enumerate(obs_dict.get("enemy", [])):
        if enemy.get("not_observed", False):
            continue
        enemy_x, enemy_y = own_x + enemy["relative_location"][0], own_y + enemy["relative_location"][1]
        ax.scatter(enemy_x, enemy_y, color='red', marker='x', s=80)
        ax.annotate(f"Enemy {j}\n{enemy['type']}\nHP:{enemy['health']:.1f}",
                    (enemy_x, enemy_y), textcoords="offset points", xytext=(0, 5), ha='center',
                    fontsize=8, bbox=dict(facecolor='white', alpha=1.0, pad=0.3))
        x_positions.append(enemy_x)
        y_positions.append(enemy_y)

    if action == 1:
        ax.annotate("Stop", (own_x, own_y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
    elif action == 2:
        ax.annotate("North", (own_x, own_y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
        ax.arrow(own_x, own_y, 0, 0.1, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
    elif action == 3:
        ax.annotate("South", (own_x, own_y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
        ax.arrow(own_x, own_y, 0, -0.1, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
    elif action == 4:
        ax.annotate("East", (own_x, own_y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
        ax.arrow(own_x, own_y, 0.1, 0, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
    elif action == 5:
        ax.annotate("West", (own_x, own_y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
        ax.arrow(own_x, own_y, -0.1, 0, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
        
    if action >= 6:
        attack_id = action - 6
        if obs_dict.get("enemy")[attack_id].get("not_observed", False):
            ax.annotate("Useless Attack", (own_x, own_y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
        else:
            act_x, act_y = obs_dict.get("enemy")[attack_id]["relative_location"]
            ax.arrow(own_x, own_y, act_x, act_y, head_width=0.01, head_length=0.01, fc='red', ec='red')
            
    ax.margins(0.2)
    ax.set_aspect('equal', adjustable='box')
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

def get_unit_name(type, race):
    
    if race == "protoss":
        if type == 0:
            return "stalker"
        elif type == 1:
            return "zealot"
        elif type == 2:
            return "colossus"

    elif race == "terran":
        if type == 0:
            return "marine" 
        elif type == 1:
            return "marauder"
        elif type == 2:
            return "medivac"
    elif race == "zerg":
        if type == 0:
            return "zergling"
        elif type == 1:
            return "hydralisk"
        elif type == 2:
            return "mutalisk"