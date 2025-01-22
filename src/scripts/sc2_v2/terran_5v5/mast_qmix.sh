#!/bin/bash

for _ in {1..2}; do
    python ../../../main.py --config=mast_qmix --env-config=sc2_v2_terran with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    env_args.use_extended_action_masking=True use_wandb=True group_name=mast_qmix;
done
