#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=mast_vdn --env-config=sc2_v2_zerg with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=mast_vdn;
done