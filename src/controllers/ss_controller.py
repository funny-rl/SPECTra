#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th

from .basic_controller import BasicMAC

class SSMAC(BasicMAC):
    def __init__(self, scheme, groups, args, eval_args = None):
        super(SSMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1
        self.eval_args = eval_args
        self.n_actions = scheme["actions"]
        
    # Add new func
    def _get_obs_component_dim(self, test_mode):
        if not test_mode or not self.args.use_CL:
            move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (5, 5), (4, 5), 1]
        else:
            move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.eval_args.obs_component # [4, (20, 5), (19, 5), 1]  
        self.n_enemies = enemy_feats_dim[0]
        self.n_allies = ally_feats_dim[0]
        self.n_agents = self.n_allies + 1
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim)
        
    def _build_inputs(self, batch, t, test_mode):
        bs = batch.batch_size
        raw_obs = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        # assert raw_obs.shape[-1] == self._get_obs_shape()
        obs_component_dim = self._get_obs_component_dim(test_mode)
        move_feats, enemy_feats, ally_feats, own_feats = th.split(raw_obs, obs_component_dim, dim=-1)
        own_feats = th.cat((own_feats, move_feats), dim=2)
        # use the max_dim (over self, enemy and ally) to init the token layer (to support all maps)

        own_feats = own_feats.reshape(bs * self.n_agents, 1, -1)
        ally_feats = ally_feats.contiguous().view(bs * self.n_agents, self.n_allies, -1)
        enemy_feats = enemy_feats.contiguous().view(bs * self.n_agents, self.n_enemies, -1)
        
        embedding_indices = []
        if self.args.obs_agent_id:
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1) / self.n_agents)
        if self.args.obs_last_action:
            if t == 0:
                last_actions = th.full_like(batch["actions"][:, 0].squeeze(-1), -1)
                embedding_indices.append(last_actions)
            else:
                last_actions = batch["actions"][:, t - 1].squeeze(-1)
                embedding_indices.append(last_actions / len(last_actions[0]))
        return bs, own_feats, ally_feats, enemy_feats, embedding_indices

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim