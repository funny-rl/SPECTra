import argparse
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm

from modules.agents.hpns_rnn_agent import HPNS_RNNAgent
from modules.agents.ss_rnn_agent import SS_RNNAgent
from modules.agents.updet_agent import UPDeT

from modules.mixers.nmix import Mixer
from modules.mixers.ss_mixer import SSMixer

output_dir = "inference_data"
os.makedirs(output_dir, exist_ok=True)

nf_al = 8
nf_en = 7

original_batch = 8
num_iter = 10000
output_normal_actions = 6

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = []

num_entities = [[i, i] for i in range(10, 101, 15)]
models = [UPDeT, SS_RNNAgent, HPNS_RNNAgent]
total_iterations = len(num_entities) * len(models) * num_iter

with tqdm(total=total_iterations, desc="Inference Progress", unit="iteration") as pbar:
    for num_entity in num_entities:
        for model in models:
            num_enemies, num_agents = num_entity
            bs = original_batch
            batch = original_batch * num_agents
            num_ally = num_agents - 1
            input_shape = [5, (num_enemies, 5), (num_agents - 1, 5)]

            if model == HPNS_RNNAgent:
                state_component = [nf_al * num_agents, nf_en * num_enemies]
                args = argparse.Namespace(
                    name="hpn",
                    n_agents=num_agents,
                    n_allies=(num_agents - 1),
                    n_enemies=num_enemies,
                    n_actions=(num_enemies + output_normal_actions),
                    hpn_head_num=1,
                    rnn_hidden_dim=64,
                    hpn_hyper_dim=64,
                    hypernet_embed=32,
                    output_normal_actions=6,
                    state_shape=(state_component[0] + state_component[1]),
                    mixing_embed_dim=32,
                    obs_agent_id=False,
                    obs_last_action=False,
                    map_type="default",
                )
                own_feats = th.rand(batch, 1, input_shape[0]).reshape(bs * num_agents, input_shape[0]).to(device)
                enemy_feats = th.rand(batch, num_enemies, input_shape[1][1]).reshape(bs * num_agents * num_enemies, input_shape[1][1]).to(device)
                ally_feats = th.rand(batch, num_ally, input_shape[2][1]).reshape(bs * num_agents * num_ally, input_shape[2][1]).to(device)
                inputs = [bs, own_feats, enemy_feats, ally_feats, None]
                hidden_state = th.rand(bs, num_agents, args.rnn_hidden_dim).to(device)

            elif model == SS_RNNAgent:
                args = argparse.Namespace(
                    name="ss",
                    n_agents=num_agents,
                    n_allies=(num_agents - 1),
                    n_enemies=num_enemies,
                    n_actions=(num_enemies + output_normal_actions),
                    n_head=4,
                    hidden_size=64,
                    rnn_hidden_dim=64,
                    output_normal_actions=6,
                    mixing_embed_dim=32,
                    mixing_n_head=1,
                    env_args={},
                    use_sqca=True,
                    obs_agent_id=False,
                    obs_last_action=False,
                    map_type="default",
                    use_SAQA=True,
                    env="sc2",
                )
                args.env_args["use_extended_action_masking"] = False
                own_feats = th.rand(batch, 1, input_shape[0]).to(device)
                enemy_feats = th.rand(batch, num_enemies, input_shape[1][1]).to(device)
                ally_feats = th.rand(batch, num_ally, input_shape[2][1]).to(device)
                inputs = [bs, own_feats, ally_feats, enemy_feats, None]
                hidden_state = th.rand(bs, num_agents, args.rnn_hidden_dim).to(device)

            elif model == UPDeT:
                state_component = [nf_al * num_agents, nf_en * num_enemies]
                args = argparse.Namespace(
                    name="updet",
                    n_agents=num_agents,
                    n_allies=(num_agents - 1),
                    n_enemies=num_enemies,
                    n_actions=(num_enemies + output_normal_actions),
                    n_head=4,
                    transformer_embed_dim=32,
                    transformer_heads=3,
                    transformer_depth=2,
                    hypernet_embed=64,
                    state_shape=(state_component[0] + state_component[1]),
                    mixing_embed_dim=32,
                    obs_agent_id=False,
                    obs_last_action=False,
                    map_type="default",
                    check_model_attention_map=False,
                    env="sc2",
                )
                own_feats = th.rand(batch, 1, input_shape[0]).to(device)
                enemy_feats = th.rand(batch, num_enemies, input_shape[1][1]).to(device)
                ally_feats = th.rand(batch, num_ally, input_shape[2][1]).to(device)
                inputs = th.cat([own_feats, enemy_feats, ally_feats], dim=1)
                hidden_state = th.rand(bs * num_agents, 1, args.transformer_embed_dim).to(device)

            vdn = model(input_shape, args).to(device)
            
            for _ in range(3):  
                with th.no_grad():
                    actions = vdn(inputs, hidden_state)

            for iter_id in range(num_iter):
                start_time = time.perf_counter()
                with th.no_grad():
                    acts, hid = vdn(inputs, hidden_state)
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000

                results.append({
                    "Model": model.__name__,
                    "Num_Agents": num_agents,
                    "Num_Enemies": num_enemies,
                    "Iteration": iter_id + 1,
                    "Elapsed_MS": elapsed_ms
                })

                del acts, hid
                th.cuda.empty_cache()
                th.cuda.synchronize()

                pbar.update(1) 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_file = os.path.join(output_dir, "inference_times.xlsx")
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")

# Plotting
plt.figure(figsize=(12, 6))
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    grouped = subset.groupby("Num_Agents")["Elapsed_MS"]

    avg_times = grouped.mean()
    std_times = grouped.std()
    min_times = grouped.min()
    max_times = grouped.max()

    lower_bound = np.clip(avg_times - std_times, min_times, max_times)
    upper_bound = np.clip(avg_times + std_times, min_times, max_times)

    plt.plot(avg_times.index, avg_times, marker="o", label=f"{model}")
    plt.fill_between(avg_times.index, lower_bound, upper_bound, alpha=0.2)

plt.title("Inference Time Comparison by Model (GPU Accelerated)")
plt.xlabel("Number of Agents")
plt.ylabel("Average Elapsed Time (ms) ± Std Dev")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.ylim(bottom=0)

plot_file = os.path.join(output_dir, "inference_times.png")
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")

plt.close()