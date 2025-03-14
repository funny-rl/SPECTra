#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch as th

from .basic_controller import BasicMAC

class SPECTraMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_enemies = args.n_enemies
        self.n_allies = args.n_allies 
        self.obs_own_feature_dims, self.obs_ball_feature_dims, \
                self.obs_ally_feature_dim, self.obs_enemy_feature_dim = self.args.obs_component
        self.observer_feats = self.obs_own_feature_dims + self.obs_ball_feature_dims
        
        super(SPECTraMAC, self).__init__(scheme, groups, args)

    def _get_obs_component_dim(self):
        return (self.observer_feats, self.obs_ally_feature_dim * self.n_allies, self.obs_enemy_feature_dim * self.n_enemies)
    
    def _build_inputs(self, batch, t, test_mode):
        bs = batch.batch_size
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        observer_feats_t, ally_feats_t, enemy_feats_t = th.split(raw_obs_t, self._get_obs_component_dim(), dim=-1)
        
        observer_feats_t = observer_feats_t.reshape(bs * self.n_agents, 1, -1)
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents, self.n_allies, -1)  
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents, self.n_enemies, -1) 
        return bs, observer_feats_t, ally_feats_t, enemy_feats_t
    
    def supervised_build_inputs(self, batch, test_mode):
        bs = batch.shape[0]
        observer_feats_t, ally_feats_t, enemy_feats_t = th.split(batch, self._get_obs_component_dim(), dim=-1)
        observer_feats_t = observer_feats_t.reshape(bs * self.n_agents, 1, -1)
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents, self.n_allies, -1)  
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents, self.n_enemies, -1) 
        return bs, observer_feats_t, ally_feats_t, enemy_feats_t

    def _get_input_shape(self, scheme):
        return ((1, self.observer_feats), (self.n_allies, self.obs_ally_feature_dim), (self.n_enemies, self.obs_enemy_feature_dim))