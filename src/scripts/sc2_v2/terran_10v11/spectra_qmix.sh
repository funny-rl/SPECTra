#!/bin/bash


for _ in {1}; do
    CUDA_VISIBLE_DEVICES="5" python ../../../main.py --config=ss_qmix --env-config=sc2_v2_terran with \
    env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=11 \
    env_args.use_extended_action_masking=True use_wandb=True group_name=spectra_qmix mixer=qmix batch_size=32  hypernet_embed=64;
done