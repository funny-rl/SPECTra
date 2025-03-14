#!/bin/bash


for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=ss_vdn --env-config=sc2_v2_protoss with \
    env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=10 \
    env_args.use_extended_action_masking=False use_wandb=True group_name=spectra_vdn batch_size=32;
done