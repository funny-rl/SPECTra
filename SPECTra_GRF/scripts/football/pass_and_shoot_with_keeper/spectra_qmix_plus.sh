#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="3" python ../../../main.py --config=spectra_qmix_plus --env-config=pass_and_shoot_with_keeper \
    with use_wandb=True group_name=SPECTra_QMIX+ local_results_path=../../../../results;
done