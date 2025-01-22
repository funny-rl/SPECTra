#!/bin/bash

for _ in {3}; do
    python ../../../main.py --config=hpn_qmix --env-config=sc2_v2_zerg with \
    env_args.use_extended_action_masking=False use_wandb=True group_name=hpn_qmix;
done