#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=ss_vdn --env-config=sc2_v2_protoss with \
    env_args.use_extended_action_masking=False use_wandb=True group_name=sa_spf_vdn use_O2MA=False;
done