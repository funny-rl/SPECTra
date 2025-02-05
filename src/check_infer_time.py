import argparse
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import sys

from modules.agents.hpns_rnn_agent import HPNS_RNNAgent
from modules.agents.ss_rnn_agent import SS_RNNAgent
from modules.agents.updet_agent import UPDeT

from modules.mixers.nmix import Mixer
from modules.mixers.ss_mixer import SSMixer

use_nonliear_mixer = False

nf_al = 8
nf_en = 7

original_batch = 1
num_iter = 1000
output_normal_actions = 6

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = []

for num_entity in [[i, i] for i in range(20, 201, 20)]:
    for model in [UPDeT, SS_RNNAgent, HPNS_RNNAgent]:  # Include all models
        num_enemies, num_agents = num_entity
        bs = original_batch
        batch = original_batch * num_agents
        num_ally = num_agents - 1
        input_shape = [5, (num_enemies, 5), (num_agents - 1, 5)]
        
        # Configure model-specific args and inputs
        if model == HPNS_RNNAgent:
            state_component = [nf_al * num_agents , nf_en * num_enemies]
            args = argparse.Namespace(
                name = "hpn",
                n_agents=num_agents,
                n_allies=(num_agents - 1),
                n_enemies=num_enemies,
                n_actions=(num_enemies + output_normal_actions),
                hpn_head_num=1,
                rnn_hidden_dim=64,
                hpn_hyper_dim=64,
                hypernet_embed = 32,
                output_normal_actions=6,
                state_shape = (state_component[0] + state_component[1]),
                mixing_embed_dim = 32,
                obs_agent_id=False,
                obs_last_action=False,
                map_type="default",
            )
            own_feats = th.rand(batch, 1, input_shape[0]).reshape(bs * num_agents, input_shape[0]).to(device)
            enemy_feats = th.rand(batch, num_enemies, input_shape[1][1]).reshape(bs * num_agents * num_enemies, input_shape[1][1]).to(device)
            ally_feats = th.rand(batch, num_ally, input_shape[2][1]).reshape(bs * num_agents * num_ally, input_shape[2][1]).to(device)
            inputs = [bs, own_feats, enemy_feats, ally_feats, None]
            hidden_state = th.rand(bs, num_agents, args.rnn_hidden_dim).to(device)
            if use_nonliear_mixer:
                vdn_output = th.rand(bs, 1, num_agents)
                _state = th.rand(bs, 1, args.state_shape)
                _mixer = Mixer(args)
            
        elif model == SS_RNNAgent:
            args = argparse.Namespace(
                name = "ss",
                n_agents=num_agents,
                n_allies=(num_agents - 1),
                n_enemies=num_enemies,
                n_actions=(num_enemies + output_normal_actions),
                n_head=4,
                hidden_size=64,
                rnn_hidden_dim=64,
                output_normal_actions=6,
                mixing_embed_dim=32,
                mixing_n_head = 1,
                env_args={},
                use_sqca=True,
                obs_agent_id=False,
                obs_last_action=False,
                map_type="default",
                env="sc2",
            )
            args.env_args["use_extended_action_masking"] = False
            own_feats = th.rand(batch, 1, input_shape[0]).to(device)
            enemy_feats = th.rand(batch, num_enemies, input_shape[1][1]).to(device)
            ally_feats = th.rand(batch, num_ally, input_shape[2][1]).to(device)
            inputs = [bs, own_feats, ally_feats, enemy_feats]
            hidden_state = th.rand(bs, num_agents, args.rnn_hidden_dim).to(device)
            args.state_component = [nf_al * num_agents , nf_en * num_enemies]
            state_dim = args.state_component[0] + args.state_component[1]
            if use_nonliear_mixer:
                
                vdn_output = th.rand(bs, 1, num_agents)
                _state = th.rand(bs, 1, state_dim)
                _mixer = SSMixer(args)
            
        elif model == UPDeT:
            state_component = [nf_al * num_agents , nf_en * num_enemies]
            args = argparse.Namespace(
                name = "updet",
                n_agents=num_agents,
                n_allies=(num_agents - 1),
                n_enemies=num_enemies,
                n_actions=(num_enemies + output_normal_actions),
                n_head=4,
                transformer_embed_dim=32,
                transformer_heads=3,
                transformer_depth=2,
                hypernet_embed= 64,
                state_shape = (state_component[0] + state_component[1]),
                mixing_embed_dim=32,
                obs_agent_id=False,
                obs_last_action=False,
                map_type="default",
                env="sc2",
            )
            token_dim = max([input_shape[0], input_shape[1][-1], input_shape[2][-1]])
            
            def zero_padding(features, token_dim):
                existing_dim = features.shape[-1]
                if existing_dim < token_dim:
                    return F.pad(features, pad=[0, token_dim - existing_dim], mode='constant', value=0)
                else:
                    return features
            
            own_feats = th.rand(batch, 1, input_shape[0]).to(device)
            enemy_feats = th.rand(batch, num_enemies, input_shape[1][1]).to(device)
            ally_feats = th.rand(batch, num_ally, input_shape[2][1]).to(device)
            inputs = th.cat([
                zero_padding(own_feats, token_dim),
                zero_padding(enemy_feats, token_dim),
                zero_padding(ally_feats, token_dim)
            ], dim=1)
            hidden_state = th.rand(bs * num_agents, 1, args.transformer_embed_dim).to(device)
            state_dim = args.state_shape
            
            if use_nonliear_mixer:
                vdn_output = th.rand(bs, 1, num_agents)
                _state = th.rand(bs, 1, state_dim)
                _mixer = Mixer(args)
                
        if use_nonliear_mixer:
            args.name = args.name + "_qmix"
        else:
            args.name = args.name + "_vdn"
            
        vdn = model(input_shape, args).to(device)
        
        for _ in range(5):  
            with th.no_grad():
                actions = vdn(inputs, hidden_state)
        print(f"Model {model.__name__} with {num_agents} agents and {num_enemies} enemies is ready for inference.")
        
        for iter_id in range(num_iter):
            start_time = time.perf_counter()
            with th.no_grad():
                acts, hid = vdn(inputs, hidden_state)
                if use_nonliear_mixer:
                    joint_q = _mixer(vdn_output, _state)

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            results.append({
                "Model": model.__name__,
                "Num_Agents": num_agents,
                "Num_Enemies": num_enemies,
                "Iteration": iter_id + 1,
                "Elapsed_MS": elapsed_ms
            })
            if use_nonliear_mixer:
                del acts, hid, joint_q
            else:
                del acts, hid
            th.cuda.empty_cache()
            th.cuda.synchronize()
            
df = pd.DataFrame(results)

output_file = "inference_times.xlsx"
df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")

plt.figure(figsize=(12, 6))
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    grouped = subset.groupby("Num_Agents")["Elapsed_MS"]
    
    avg_times = grouped.mean()
    std_times = grouped.std()  # 표준 편차 계산
    
    plt.plot(avg_times.index, avg_times.values, marker="o", label=f"{model}")
    plt.fill_between(avg_times.index, avg_times - std_times, avg_times + std_times, alpha=0.2)

plt.title("Inference Time Comparison by Model (GPU Accelerated)")
plt.xlabel("Number of Agents")
plt.ylabel("Average Elapsed Time (ms) ± Std Dev")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()

plot_file = "inference_times_with_variance.png"
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")

plt.close()