#!/bin/bash

# SQCA -> self attention 

for _ in {1..2}; do
    CUDA_VISIBLE_DEVICES="3" python ../../main.py --config=ss_qmix --env-config=academy_pass_and_shoot_with_keeper with use_wandb=True group_name=spectra_qmix+;
done