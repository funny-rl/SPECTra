import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.ss_attention import QueryKeyBlock, CrossAttentionBlock, PoolingQueryKeyBlock

class Hypernetwork(nn.Module):
    def __init__(self, args, input_shape, last_action = False):
        self.args = args
        super(Hypernetwork, self).__init__()
        self.n_head = args.mixing_n_head
        self.hypernet_embed = args.hypernet_embed
        self.input_shape = input_shape
        
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.entities = self.n_agents + self.n_enemies
        
        
        self.agent_embedding = nn.Linear(self.input_shape[0], self.hypernet_embed)
        self.enemy_embedding = nn.Linear(self.input_shape[1], self.hypernet_embed)
        
        self.agent_features = self.input_shape[0] * self.n_agents
        self.enemy_features = self.input_shape[1] * self.n_enemies

        self.add_embed = False
        if self.args.env_args["state_last_action"] and last_action:
            self.hypernet_embed = self.hypernet_embed + self.input_shape[2]
            self.add_embed = True
            
        self.cross_attention = CrossAttentionBlock(
            d = self.hypernet_embed,
            h = self.n_head,
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
        
        agent_state = state[:, :self.agent_features].reshape(bs_t, self.n_agents, -1)
        enemy_state = state[:, self.agent_features : self.agent_features + self.enemy_features].reshape(bs_t, self.n_enemies, -1)
        
        a_embed = self.agent_embedding(agent_state)
        e_embed = self.enemy_embedding(enemy_state)
        
        if self.add_embed:
            action_state = state[:, self.agent_features + self.enemy_features:].reshape(bs_t, self.n_agents, -1)
            e_padding = th.zeros(e_embed.size(0), e_embed.size(1), action_state.size(-1)).to(device=e_embed.device)
            
            e_embed = th.cat((e_embed, e_padding), dim=-1)
            a_embed = th.cat((a_embed, action_state), dim=-1)

        embed = th.cat((a_embed, e_embed), dim=1)
        x = self.cross_attention(a_embed, embed)
        
        weight = self.weight_generator(x, x)
        bias = self.bias_generator(x)
        return weight, bias 
    
class SSMixer(nn.Module):
    def __init__(self, args, abs = True):
        super(SSMixer, self).__init__()
        
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1
        
        self.state_component = args.state_component
        self.state_shape, state_feature_dims = self._get_input_shape(state_component = self.state_component)
        self.state_dim = sum(self.state_component)
        
        # hyper w1 b1
        self.hyper_w1 = Hypernetwork(
            args = args, 
            input_shape = state_feature_dims,
        )

        self.hyper_w2 = Hypernetwork(
            args = args, 
            input_shape = state_feature_dims,
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
        

    def _get_input_shape(self, state_component):
        entity_type = [self.n_agents, self.n_enemies, self.n_agents, 1] # U have to change this when u change your state entity sequence
        state_shape = []
        state_feature_dims = []
        for idx, component in enumerate(state_component):
            state_shape.append((entity_type[idx], int(component / entity_type[idx])))
            state_feature_dims.append(int(component / entity_type[idx]))
        return state_shape, state_feature_dims
        
