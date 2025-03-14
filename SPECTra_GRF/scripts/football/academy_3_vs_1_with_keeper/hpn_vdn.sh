#!/bin/bash

# SQCA -> self attention 

for _ in {1..3}; do
    CUDA_VISIBLE_DEVICES="4" python ../../main.py --config=hpn_vdn --env-config=academy_3_vs_1_with_keeper with use_wandb=True group_name=hpn_vdn;
done