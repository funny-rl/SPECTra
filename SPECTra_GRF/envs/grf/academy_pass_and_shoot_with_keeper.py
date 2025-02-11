from .. import MultiAgentEnv
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np

from .utils import _get_avail, Rewarder

class Academy_Pass_and_Shoot_with_Keeper(MultiAgentEnv):
    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=1000,
        render=False,
        n_agents=2,
        n_enemies=2,
        time_limit=150,
        time_step=0,
        obs_dim=22,
        env_name='academy_pass_and_shoot_with_keeper',
        stacked=False,
        representation="simple115",
        rewards='scoring',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0
    ):

        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.episode_limit = time_limit
        self.time_step = time_step
        self.obs_dim = obs_dim
        self.env_name = env_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed
        
        self.reward_encoder = Rewarder(self.n_agents)
        
        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n


        self.unit_dim = self.obs_dim  # QPLEX unit_dim for cds_gfootball
        # self.unit_dim = 8  # QPLEX unit_dim set like that in Starcraft II
        
    def get_simple_obs(self, index=-1):
        
        simple_obs = []
        if index == -1:
            full_obs = self.env.unwrapped.observation()[0]
            for idx, agents in enumerate(full_obs['left_team'][-self.n_agents:]):
                simple_obs.append(agents.reshape(-1))
                simple_obs.append(
                    full_obs['left_team_direction'][-self.n_agents:][idx].reshape(-1))
            for idx, enemies in enumerate(full_obs['right_team']):
                simple_obs.append(enemies.reshape(-1))
                simple_obs.append(full_obs['right_team_direction'][idx].reshape(-1))
            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

            simple_obs = np.concatenate(simple_obs)
            return simple_obs

        else:
            full_obs = self.env.unwrapped.observation()[index]
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))

            
            for idx, xy_coor in enumerate(np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0)):
                simple_obs.append((xy_coor- ego_position).reshape(-1))
                simple_obs.append(np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0)[idx].reshape(-1))
                
            # simple_obs.append((np.delete(
            #     full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))
            # simple_obs.append(np.delete( 
            #     full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            for idx, xy_coor in enumerate(full_obs['right_team']):
                simple_obs.append((xy_coor- ego_position).reshape(-1))
                simple_obs.append(full_obs['right_team_direction'][idx].reshape(-1))
                
            # simple_obs.append(
            #     (full_obs['right_team'] - ego_position).reshape(-1))
            # simple_obs.append(full_obs['right_team_direction'].reshape(-1))
            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

            simple_obs = np.concatenate(simple_obs)
            return simple_obs, full_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        self._obs, original_rewards, done, infos = self.env.step(actions.tolist())
        
        rewards = self.reward_encoder.calc_reward(original_rewards[0], self.pre_obs[0], self._obs[0])
        
        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])
        
        if done:
            lose = True

        if self.time_step >= self.episode_limit:
            done = True
            lose = True

        if self.check_if_done():
            done = True
            lose = True
            
        if sum(original_rewards) > 0.0:
            lose = False

        self.pre_obs = self._obs

        if done:
            if lose:
                # return obs, self.get_global_state(), -int(done), done, infos
                return rewards - 1, done, infos
            else:
                return rewards + 100, done, infos

        else:
            return rewards, done, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self.get_obs_agent()

    def avail_action_mask(self, full_obses):
        self.avail_actions = []
        for full_obs in full_obses:
            player_idx = full_obs["active"]
            player_pos_x, player_pos_y = full_obs["left_team"][player_idx]
            ball_x, ball_y, _ = full_obs["ball"]
            ball_x_relative = ball_x - player_pos_x
            ball_y_relative = ball_y - player_pos_y
            ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
            self.avail_actions.append(_get_avail(full_obs, ball_distance))
            

    def get_obs_agent(self):
        """Returns observation for agent_id."""
        obs = []
        self.full_obs = []
        for i in range(self.n_agents):
            simple_obs, full_obs = self.get_simple_obs(i)
            obs.append(simple_obs)
            self.full_obs.append(full_obs)
        self.avail_action_mask(full_obses = self.full_obs)
        return obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return self.avail_actions
    
    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.pre_obs = self.env.reset()
        
        self._obs = self.pre_obs
    
        return np.array(self.get_obs_agent()), self.get_global_state()
    
    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

