import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.spectra_attention import QueryKeyBlock, CrossAttentionBlock, PoolingQueryKeyBlock

class ST_HyperNet(nn.Module):
    def __init__(self, args):
        self.args = args
        super(ST_HyperNet, self).__init__()
        self.n_head = args.mixing_n_head
        self.hypernet_embed = args.hypernet_embed
        
        self.n_agents = args.n_agents
        self.n_players = self.n_agents + 1  # including goalkeeper
        self.n_enemies = args.n_enemies
        self.entities = self.n_agents + self.n_enemies
        
        self.state_component = args.state_component
        
        self.agent_embedding = nn.Linear(self.state_component[0], self.hypernet_embed)
        self.enemy_embedding = nn.Linear(self.state_component[1], self.hypernet_embed)
        self.ball_embedding = nn.Linear(self.state_component[2], self.hypernet_embed)
        
        self.agent_features = self.state_component[0] * self.n_players
        self.enemy_features = self.state_component[1] * self.n_enemies
            
        self.cross_attention = CrossAttentionBlock(
            d = self.hypernet_embed,
            h = self.n_head,
        )
        
        self.weight_mlp = nn.Sequential(
            nn.Linear(self.hypernet_embed, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.hypernet_embed)
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.hypernet_embed, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.hypernet_embed)
        )
        
        self.weight_generator = QueryKeyBlock(
            d = self.hypernet_embed, 
            h = self.n_head
        )
        
        self.bias_generator = PoolingQueryKeyBlock(
            d = self.hypernet_embed,
            k = 1,
            h = self.n_head 
        )

    def forward(self, state): # state: [batch * t, state] 
        bs_t = state.size(0)
        
        agent_state = state[:, :self.agent_features].reshape(bs_t, self.n_players, -1)
        enemy_state = state[:, self.agent_features : self.agent_features + self.enemy_features].reshape(bs_t, self.n_enemies, -1)
        ball_state = state[:, self.agent_features + self.enemy_features:].reshape(bs_t, 1, -1)

        a_embed = self.agent_embedding(agent_state)
        e_embed = self.enemy_embedding(enemy_state)
        b_embed = self.ball_embedding(ball_state)

        embed = th.cat((a_embed, e_embed, b_embed), dim=1)
        x = self.cross_attention(a_embed, embed)
        weight_x = self.weight_mlp(x)
        bias_x = self.bias_mlp(x)
        weight_x = (x + weight_x)[:, 1:]
        bias_x = (x + bias_x)[:, 1:]
        
        weight = self.weight_generator(weight_x, weight_x)
        bias = self.bias_generator(bias_x)
        return weight, bias 
    
class SPECTraMixer(nn.Module):
    def __init__(self, args, abs = True):
        super(SPECTraMixer, self).__init__()
        
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        
        self.hyper_w1 = ST_HyperNet(
            args = args, 
        )

        self.hyper_w2 = ST_HyperNet(
            args = args, 
        )

        self.abs = abs # monotonicity constraint

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(b * t, -1)
        
        # First layer
        w1, b1 = self.hyper_w1(states)
        # Second layer
        w2, b2 = self.hyper_w2(states)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        h1 = F.elu(th.matmul(qvals, w1) + b1)
        
        h2 = (th.matmul(h1, w2) + b2).sum(dim=-1, keepdim=False) 
        return h2.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
    