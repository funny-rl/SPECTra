#!/bin/bash

# SQCA -> self attention 

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="5" python ../../main.py --config=hpn_vdn --env-config=academy_run_pass_and_shoot_with_keeper with use_wandb=True group_name=hpn_vdn;
done