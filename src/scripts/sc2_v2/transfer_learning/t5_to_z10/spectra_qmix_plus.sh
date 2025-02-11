#!/bin/bash

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../../main.py --config=ss_qmix --env-config=sc2_v2_zerg with \
    env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=10 \
    use_wandb=True group_name=spectra_qmix+_t5z10 batch_size=32 transfer_learning=True \
    load_dir="../../../../models/terran_5v5/ss_qmix" t_max=5000000;
done