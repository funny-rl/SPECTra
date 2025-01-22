#!/bin/bash

for _ in {1..2}; do
    python ../../../main.py --config=mast_vdn --env-config=sc2_v2_zerg with \
    env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=10 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=mast_vdn;
done