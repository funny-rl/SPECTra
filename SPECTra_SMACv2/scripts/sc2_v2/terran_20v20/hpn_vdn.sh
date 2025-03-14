#!/bin/bash

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=hpn_vdn --env-config=sc2_v2_terran with \
    env_args.capability_config.n_units=20 env_args.capability_config.n_enemies=20 \
    batch_size=10 env_args.use_extended_action_masking=False use_wandb=True group_name=hpn_vdn;
done