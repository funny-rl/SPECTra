import numpy as np 
import torch as th

def build_td_lambda_targets(args, rewards, terminated, mask, target_qs, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    """
    target_qs  : batch_size x (episode_length + 1) x 1
    rewards    : batch_size x episode_length x 1 : after terminated all are 0
    terminated : batch_size x episode_length x 1 : only have a one 1 other are 0
    mask       : batch_size x episode_length x 1 : before terminated all are 1, after terminated all are 0
    """
    n_reward = args.n_reward
    if  n_reward == 1:
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
        for t in range(ret.shape[1] - 2, -1,  -1):
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                        * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    else:
        gamma = th.tensor(np.array(args.gamma_set)).to(args.device) # list of gamma values
        gamma_sum = gamma.sum()
        gamma_rate = gamma / gamma_sum
        target_qs = target_qs.repeat(1, 1, n_reward)
        for idx, gamma_value in enumerate(gamma_rate):
            target_qs[:, :, idx] *= gamma_value
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1)) # if episode is terminated then ret[:, -1] is 0.
        # Backwards  recursive  update  of the "forward  view"
        for t in range(ret.shape[1] - 2, -1,  -1): 
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + \
                mask[:, t] * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
        ret = th.sum(ret, dim=2, keepdim=True)
        # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def build_q_lambda_targets(rewards, terminated, mask, exp_qvals, qvals, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = exp_qvals.new_zeros(*exp_qvals.shape)
    ret[:, -1] = exp_qvals[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        reward = rewards[:, t] + exp_qvals[:, t] - qvals[:, t] #off-policy correction
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (reward + (1 - td_lambda) * gamma * exp_qvals[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]