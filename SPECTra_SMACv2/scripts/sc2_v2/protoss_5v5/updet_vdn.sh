#!/bin/bash


for _ in {1}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=updet_vdn --env-config=sc2_v2_protoss with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    use_wandb=True group_name=updet_vdn;
done