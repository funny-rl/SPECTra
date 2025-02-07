#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="3" python ../../../main.py --config=ss_vdn --env-config=sc2_v2_terran with \
    env_args.use_extended_action_masking=False use_wandb=True group_name=unex_spf_vdn;
done