#!/bin/bash

for _ in {1}; do
    CUDA_VISIBLE_DEVICES="3" python ../../../main.py --config=spectra_vdn --env-config=academy_3_vs_1_with_keeper \
    with use_wandb=True group_name=SPECTra_VDN local_results_path=../../../../results;
done
