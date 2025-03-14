from .. import MultiAgentEnv

import gym
import numpy as np
import gfootball.env as football_env

from gym.spaces import Box
from gfootball.env import observation_preprocessing

from .utils import FeatureModel, RewardModel

class Run_pass_and_shoot_with_keeper(MultiAgentEnv):
    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=3000,
        render=False,
        n_agents = 2,
        n_enemies = 2,
        episode_limit=400,
        time_step=0,
        env_name='academy_3_vs_1_with_keeper',
        stacked=False,
        representation="raw",
        rewards='scoring',
        logdir="../results/replays",
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0,
        batch_size_run= 8,
    ):
        self.n_enemies = n_enemies
        self.dense_reward = dense_reward
        self.n_agents = n_agents
        self.n_allies = n_agents # because of goal keeper
        self.episode_limit = episode_limit
        self.time_step = time_step
        self.write_video = write_video
        self.seed = seed
        
        self.env = football_env.create_environment(
            write_full_episode_dumps=write_full_episode_dumps,
            write_goal_dumps = write_goal_dumps,
            env_name = env_name,
            stacked = stacked,
            representation = representation,
            rewards = rewards,
            logdir = logdir,
            render = render,
            write_video = write_video,
            dump_frequency = dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls = number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        
        self.obs_own_feature_dims = 21
        self.obs_ball_feature_dims = 18
        self.obs_ally_feature_dim = 16
        self.obs_enemy_feature_dim = 16

        self.state_agent_feature_dim = 18
        self.state_enemy_feature_dim = 17
        self.state_ball_feature_dims = 15
        
        self.feature_encoder = FeatureModel(
            n_agents = self.n_agents, 
            n_enemies = self.n_enemies,
            obs_ally_feature_dim = self.obs_ally_feature_dim,
            obs_enemy_feature_dim = self.obs_enemy_feature_dim,
            state_agent_feature_dim = self.state_agent_feature_dim,
            state_enemy_feature_dim = self.state_enemy_feature_dim,
        )
        self.reward_model = RewardModel(
            episode_limit = self.episode_limit, 
            batch_size_run = batch_size_run
        )
        
        # init action space
        self.action_space = [gym.spaces.Discrete(self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        
        # init observation & state
        init_observations = self.env.reset() # List of observation dictionaries.
        
        tmp_obs = [self._encode_obs(init_observation)[0] for init_observation in init_observations]
        _ervation_space = [Box(low=float("-inf"), high=float("inf"), shape=tmp_obs[n].shape, dtype=np.float32)
                                  for n in range(self.n_agents)]
        self.share_observation_space = _ervation_space.copy()
        self.obs_dim = len(tmp_obs[0])
        self.state_dim = self._encode_state(init_observations[0]).shape[-1]
        
        self.n_actions = self.action_space[0].n
        self.pre_obs = None

    def _encode_obs(self, raw_obs):
        obs, ava = self.feature_encoder.encode(raw_obs.copy())
        obs_cat = np.hstack(
            [np.array(obs[k], dtype=np.float32).flatten() for k in obs]
        )
        return obs_cat, ava
    
    def _encode_state(self, raw_obs):
        return self.feature_encoder.encode_state(raw_obs.copy())

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def step(self, actions, batch_idx):
        """Returns reward, terminated, info."""
        self.time_step += 1
        o, _, done, infos  = self.env.step(actions.tolist())
        
        if self.time_step >= self.episode_limit:
            done = True
        if self.check_if_done():
            done = True
        
        obs = []
        ava = []
        for obs_dict in o:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
        
        self._obs = np.array(obs)
        self._state = self._encode_state(obs_dict)  # Because it is absolute coordinates, it does not matter if information from a specific agent is used.
        self.ava = np.array(ava)
        
        own_changing_r, oob_r, pass_r, player_pos_r, yellow_r, ball_position_r, score_r = self.reward_model.calc_reward(
            self.pre_obs[0], 
            o[0], 
            done, 
            actions, 
            batch_idx,
            self.time_step
        )
        
        rewards = (own_changing_r, oob_r, pass_r, player_pos_r, yellow_r, ball_position_r, score_r)

        self.pre_obs = o
        return rewards, done, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self._obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self._state

    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.state_dim
    
    def get_obs_component(self):
        return [self.obs_own_feature_dims, self.obs_ball_feature_dims, self.obs_ally_feature_dim, self.obs_enemy_feature_dim]
    
    def get_state_component(self):
        return [self.state_agent_feature_dim, self.state_enemy_feature_dim, self.state_ball_feature_dims]

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return self.ava
    
    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n
    
    def reset(self):
        """ Returns initial observations and states"""
        self.time_step = 0
        obs_dicts = self.env.reset()
        self.pre_obs = obs_dicts
        obs = []
        ava = []
        for obs_dict in obs_dicts:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
            
        self._obs = np.array(obs)
        self._state = self._encode_state(obs_dicts[0])
        self.ava = np.array(ava)
        return self._obs, self._state
    
    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

