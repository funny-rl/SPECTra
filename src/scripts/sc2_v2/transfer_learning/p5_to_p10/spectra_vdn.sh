#!/bin/bash

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../../main.py --config=ss_vdn --env-config=sc2_v2_protoss with \
    env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 \
    use_wandb=True group_name=spectra_vdn_p5p10 batch_size=32 transfer_learning=True \
    load_dir="../../../../models/protoss_5v5/ss_vdn"  t_max=5000000 save_model_interval=2400000;
done