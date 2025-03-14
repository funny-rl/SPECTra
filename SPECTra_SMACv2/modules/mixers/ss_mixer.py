import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.ss_attention import QueryKeyBlock, CrossAttentionBlock, PoolingQueryKeyBlock

class Hypernetwork(nn.Module):
    def __init__(self, args, input_shape):
        self.args = args
        super(Hypernetwork, self).__init__()
        self.n_head = args.mixing_n_head
        self.hypernet_embed = args.hypernet_embed
        self.state_last_action = self.args.env_args["state_last_action"]
        self.state_timestep_number = self.args.env_args["state_timestep_number"]
        self.input_shape = input_shape
        
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.entities = self.n_agents + self.n_enemies
        
        self.additional_input_dim = sum([self.state_last_action, self.state_timestep_number])
        self.agent_embedding = nn.Linear(self.input_shape[0] + self.additional_input_dim, self.hypernet_embed)
        self.enemy_embedding = nn.Linear(self.input_shape[1], self.hypernet_embed)
        
        self.agent_features = self.input_shape[0] * self.n_agents
        self.enemy_features = self.input_shape[1] * self.n_enemies
        self.entity_features = self.agent_features + self.enemy_features
        
        self.n_actions = self.input_shape[2]
        
        if self.state_last_action:
            self.action_features = self.input_shape[2] * self.n_agents 
            
        self.cross_attention = CrossAttentionBlock(
            d = self.hypernet_embed,
            h = self.n_head,
        )
        
        self.weight_embedding = nn.Sequential(
            nn.Linear(self.hypernet_embed, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.hypernet_embed),
            nn.ReLU(),
        )
        
        self.bias_embedding = nn.Sequential(
            nn.Linear(self.hypernet_embed, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.hypernet_embed),
            nn.ReLU(),
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

    def forward(self, state): # state: [batch * t, state]  # torch.Size([10624, 5, 8]) torch.Size([10624, 5, 11]) torch.Size([10624, 1, 1])
        bs_t = state.size(0)
        
        agent_state = state[:, :self.agent_features].reshape(bs_t, self.n_agents, -1)
        enemy_state = state[:, self.agent_features : self.entity_features].reshape(bs_t, self.n_enemies, -1)

        if self.state_last_action:
            last_action_states = state[:, self.entity_features: self.entity_features + self.action_features].reshape(bs_t * self.n_agents, -1)
            nonzero_indices = th.nonzero(last_action_states)
            last_actions = th.full((bs_t * self.n_agents,), -1, dtype=th.long, device=state.device)
            if nonzero_indices.numel() > 0:
                last_actions[nonzero_indices[:, 0]] = nonzero_indices[:, 1]
            state_last_actions = last_actions.reshape(bs_t, self.n_agents, -1)
            agent_state = th.cat((agent_state, state_last_actions), dim=-1)

        if self.state_timestep_number:
            timestep_state = state[:, -1].reshape(bs_t, 1, -1).repeat(1, self.n_agents, 1).to(state.device)
            agent_state = th.cat((agent_state, timestep_state), dim=-1)
        
        a_embed = self.agent_embedding(agent_state)
        e_embed = self.enemy_embedding(enemy_state)
        embed = th.cat((a_embed, e_embed), dim=1)
        x = self.cross_attention(a_embed, embed)
        
        weight_embed = self.weight_embedding(x)
        bias_embed = self.bias_embedding(x)
        
        weight = self.weight_generator(weight_embed, weight_embed)
        bias = self.bias_generator(bias_embed)
        return weight, bias 
    
class SSMixer(nn.Module):
    
    def __init__(self, args, abs = True):
        super(SSMixer, self).__init__()
        
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1
        
        self.state_component = args.state_component
        self.state_shape = self._get_input_shape(state_component = self.state_component)
        state_feature_dims = [shape[1] for shape in self.state_shape]
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
        for idx, component in enumerate(state_component):
            e_type = entity_type[idx]
            state_shape.append((e_type, int(component / e_type)))
        return state_shape
        
