import torch as th
import torch.nn as nn

from modules.layer.ss_attention import CrossAttentionBlock, QueryKeyBlock,  SetAttentionBlock

class SS_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SS_RNNAgent, self).__init__()
        self.args = args
        self.use_SAQA = self.args.use_SAQA
        self.n_actions = self.args.n_actions
        self.n_head = self.args.n_head
        self.hidden_size = self.args.hidden_size
        self.output_normal_actions = self.args.output_normal_actions
        self.use_extended_action_masking = self.args.env_args["use_extended_action_masking"]
        
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]
        
        if self.args.obs_agent_id:
            self.own_feats_dim += 1
        if self.args.obs_last_action:
            self.own_feats_dim += 1 
            
        self.own_embedding = nn.Linear(self.own_feats_dim, self.hidden_size)
        self.allies_embedding = nn.Linear(self.ally_feats_dim, self.hidden_size) 
        self.enemies_embedding = nn.Linear(self.enemy_feats_dim, self.hidden_size)
        
        self.normal_actions_net = nn.Linear(self.hidden_size, self.output_normal_actions)
        
        # Decide whether to use SQCA or self-attention
        if self.use_SAQA:
            self.single_agent_query_attention = CrossAttentionBlock(
                d = self.hidden_size, 
                h = self.n_head
            )
        else:
            self.multi_agent_query_attention = SetAttentionBlock(
                d = self.hidden_size, 
                h = self.n_head
            )
        
        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        
        self.action_attention = QueryKeyBlock(
            d = self.hidden_size, 
            h = self.n_head,
        )
        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.own_embedding.weight.new(1, self.hidden_size).zero_()
        
    
    def forward(self, inputs, hidden_state):
        """
        inputs:
            batch
            batch * self.n_agents x 1 x own_feats
            batch * self.n_agents x n_allies x ally_feats
            batch * self.n_agents x n_enemies x enemy_feats
        hidden_state: 
            batch x num_agents x hidden_size    
        """
        
        bs, own_feats, ally_feats, enemy_feats, embedding_indices  = inputs
        
        self.n_agents = ally_feats.shape[1] + 1
        
        own_mask = ~th.all(own_feats == 0, dim=-1)
        ally_mask = ~th.all(ally_feats == 0, dim=-1)
        enemy_mask = ~th.all(enemy_feats == 0, dim=-1)
        
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0].reshape(-1, 1, 1)
            own_feats = th.cat((own_feats, agent_indices), dim=-1)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1].reshape(-1, 1, 1)
            own_feats = th.cat((own_feats, last_action_indices), dim=-1)
        
        masks = th.cat((own_mask, ally_mask, enemy_mask), dim=-1).unsqueeze(1)
        own_feats = self.own_embedding(own_feats)
        ally_feats = self.allies_embedding(ally_feats)
        enemy_feats = self.enemies_embedding(enemy_feats)
        
        embeddings = th.cat((own_feats, ally_feats, enemy_feats), dim=1) # (bs * n_agents, n_entities, hidden_size)
        
        if self.use_SAQA:
            action_query = self.single_agent_query_attention(embeddings[:, 0].unsqueeze(1), embeddings, None) # (bs * n_agents, 1, hidden_size)
        else:
            action_query = self.multi_agent_query_attention(embeddings, masks.repeat(1, self.n_entities, 1))
            action_query = action_query.mean(dim = 1, keepdim = True)
        
        
        action_query = action_query.reshape(-1, self.hidden_size) # (bs * n_agents, hidden_size)
    
        hidden_state = hidden_state.reshape(-1, self.hidden_size) # (bs * n_agents, hidden_size)
        hidden_state = self.rnn(action_query, hidden_state)
        
        action_query = hidden_state.unsqueeze(1) # (bs * n_agents, 1, hidden_size)
        
        hidden_state = hidden_state.reshape(bs, self.n_agents, self.hidden_size)

        q_normal = self.normal_actions_net(action_query)
        
    
        if self.use_extended_action_masking:
            action_key = embeddings
        elif "sc2" in self.args.env:
            action_key = embeddings[:, self.n_agents:]
            
        q_interact = self.action_attention(action_query, action_key)

        return th.cat((q_normal, q_interact), dim=-1), hidden_state
    

        
        