#!/bin/bash

# SQCA -> self attention 

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="0" python ../../main.py --config=ss_vdn --env-config=academy_3_vs_1_with_keeper with use_wandb=True group_name=spectra_vdn;
done