from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from utils.th_utils import get_parameters_num

import torch.nn.functional as F
# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, id = None):
        if id is not None:
            self.id = id
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, test_mode)
        self.agent_inputs = agent_inputs
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = float('-inf')

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size, n_agents):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, n_agents, -1)  # bav

    def set_train_mode(self):
        self.agent.train()

    def set_evaluation_mode(self):
        self.agent.eval()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def get_device(self):
        return next(self.parameters()).device

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        state_dict = th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        model_dict = self.agent.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        self.agent.load_state_dict(filtered_state_dict, strict=False)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        print("&&&&&&&&&&&&&&&&&&&&&&", self.args.agent, get_parameters_num(self.parameters()))
        # for p in list(self.parameters()):
        #     print(p.shape)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    def extract_attention_score(self):
        if self.args.name == "updet_vdn":
            attention_scores = self.agent.transformer.tblocks[0].attention.attention_score
            return attention_scores
            
        elif self.args.name == "ss_vdn":
            attention_scores = self.agent.single_agent_query_attention.mab.multihead.attention.attention_score
            return attention_scores
    
    def get_agent_obs(self):
        if self.args.name == "ss_vdn":
            _, own_feats, ally_feats, enemy_feats, _ = self.agent_inputs
            max_id = max(own_feats.shape[-1], ally_feats.shape[-1], enemy_feats.shape[-1])
            outputs = th.cat([
                self.zero_padding(own_feats, max_id),
                self.zero_padding(ally_feats, max_id),
                self.zero_padding(enemy_feats, max_id),
            ], dim=1)

        if self.args.name == "updet_vdn":
            outputs = self.agent_inputs
            
        return outputs
    
    def zero_padding(self, features, token_dim):
        """
        :param features: [bs * n_agents, k, fea_dim]
        :param token_dim: maximum of fea_dim
        :return:
        """
        existing_dim = features.shape[-1]
        if existing_dim < token_dim:
            # padding to the right side of the last dimension of the feature.
            return F.pad(features, pad=[0, token_dim - existing_dim], mode='constant', value=0)
        else:
            return features