import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

import torch.autograd.profiler as profiler

def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)

class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)


class HPNS_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HPNS_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.own_feats_dim, self.ally_feats_dim, self.enemy_feats_dim  = input_shape
        self.ally_feats_dim = self.ally_feats_dim[-1]
        self.enemy_feats_dim = self.enemy_feats_dim[-1]

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK
        
        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_enemy = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, ((self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
        )  # output shape: (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)

        self.hyper_ally = nn.Sequential(
            nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, self.ally_feats_dim * self.rnn_hidden_dim * self.n_heads)
        )  # output shape: ally_feats_dim * rnn_hidden_dim

        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.fc2_normal_actions = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.rnn_hidden_dim, self.n_actions -3)
        )  

        
        self.passing_mlp =  nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.rnn_hidden_dim, 3)
        ) 

        # Reset parameters for hypernets
        self._reset_hypernet_parameters(init_type="xavier")
        self._reset_hypernet_parameters(init_type="kaiming")

    def _reset_hypernet_parameters(self, init_type='kaiming'):
        gain = 2 ** (-0.5)
        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        for m in self.hyper_enemy.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)
        for m in self.hyper_ally.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, ally_feats_t, enemy_feats_t = inputs
        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (3) Enemy feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)
        hyper_enemy_out = self.hyper_enemy(enemy_feats_t)
        fc1_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
            -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads
        )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim]
        # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
        
        
        
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

        # (4) Ally features
        hyper_ally_out = self.hyper_ally(ally_feats_t)
        # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head]
        fc1_w_ally = hyper_ally_out.view(-1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
        # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, n_heads* rnn_hidden_dim] = [bs * n_agents * n_allies, 1, n_heads*rnn_hidden_dim]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
            bs * self.n_agents, self.n_allies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, head, rnn_hidden_dim]

        passing_dim = self.passing_mlp(embedding_allies).permute(0,1,3,2).mean(dim=-1)
        passing_embed, _ = passing_dim.max(dim=1)
        
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # Final embedding
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )  # [bs * n_agents, head, rnn_hidden_dim]
        
        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]

        # Q-values of normal actions
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 16]
        passing_embed = passing_embed.view(bs, self.n_agents, -1)
        
        q = th.cat((q_normal[:, :, :9],passing_embed, q_normal[:, :, 9:] ), dim = -1)
        
        
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # [bs, n_agents, 6 + n_enemies]
