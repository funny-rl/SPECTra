#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=ss_vdn --env-config=sc2_v2_terran with \
    env_args.use_extended_action_masking=True use_wandb=True group_name=sa_spectra_vdn use_SAQA=False;
done