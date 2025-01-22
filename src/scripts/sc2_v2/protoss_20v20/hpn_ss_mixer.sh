#!/bin/bash

for _ in {1..2}; do
    python ../../../main.py --config=hpn_qmix --env-config=sc2_v2_protoss with \
    env_args.capability_config.n_units=20 env_args.capability_config.n_enemies=20 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=hpn_ss_mixer mixer=ss_mixer;
done