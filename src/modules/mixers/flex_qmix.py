import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _get_mixer_ins(self, batch, repeat_batch=1):
    if not self.args.entity_scheme:
        return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1),
                batch["state"][:, 1:])
    else:
        entities = []
        bs, max_t, ne, ed = batch["entities"].shape
        entities.append(batch["entities"])
        if self.args.entity_last_action:
            last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                    device=batch.device,
                                    dtype=batch["entities"].dtype)
            last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
            entities.append(last_actions)

        entities = th.cat(entities, dim=3)
        return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                    batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                (entities[:, 1:],
                    batch["entity_mask"][:, 1:]))

class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = int(np.prod(args.state_shape)) 
        # if self.args.env_args["state_last_action"]:
        #     self.entity_dim += args.n_actions
        # if extra_dims > 0:
        #     self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)

        self.attn = EntityAttentionLayer(hypernet_embed,
                                            hypernet_embed,
                                            hypernet_embed, args)

        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None):
        
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask)
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2), 0)
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            return x3.mean(dim=2)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3


class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionHyperNet(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = AttentionHyperNet(args, mode='scalar')

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh

    def forward(self, agent_qs, inputs):
        entities = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(entities)
        b1 = self.hyper_b_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        w1 = F.softmax(w1, dim=-1)
            
        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = F.softmax(self.hyper_w_final(entities, entity_mask), dim=-1)

        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(entities, entity_mask).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class LinearFlexQMixer(nn.Module):
    def __init__(self, args):
        super(LinearFlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='alt_vector')
        self.V = AttentionHyperNet(args, mode='scalar')

    def forward(self, agent_qs, inputs, imagine_groups=None, ret_ingroup_prop=False):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)

        agent_qs = agent_qs.view(-1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1)

        w1 = F.softmax(w1, dim=1)
        # State-dependent bias
        v = self.V(entities, entity_mask)

        q_cont = agent_qs * w1
        q_tot = q_cont.sum(dim=1) + v
        # Reshape and return
        q_tot = q_tot.view(bs, -1, 1)
        if ret_ingroup_prop:
            ingroup_w = w1.clone()
            ingroup_w[:, self.n_agents:] = 0  # zero-out out of group weights
            ingroup_prop = (ingroup_w.sum(dim=1)).mean()
            return q_tot, ingroup_prop
        return q_tot