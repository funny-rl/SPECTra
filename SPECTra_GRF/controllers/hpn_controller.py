#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()


# This multi-agent controller shares parameters between agents
class HPNMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

        self.own = 4
        self.ally = 4 * self.n_allies
        self.enemy = 4 * self.n_enemies
        self.ball = 6
        
        super(HPNMAC, self).__init__(scheme, groups, args)

    # Add new func
    def _get_obs_component_dim(self):
        return (self.own, self.ally, self.enemy, self.ball)
        
    def _build_inputs(self, batch, t, test_mode):
        bs = batch.batch_size
        
        obs_component_dim = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        own_feats_t, ally_feats_t, enemy_feats_t, ball_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]


        return bs, th.cat((own_feats_t.reshape(-1, self.own), ball_feats_t.reshape(-1, self.ball)), dim = -1), ally_feats_t, enemy_feats_t

    def _get_input_shape(self, scheme):
        return (self.own + self.ball, (self.n_allies, 4), (self.n_enemies, 4))
