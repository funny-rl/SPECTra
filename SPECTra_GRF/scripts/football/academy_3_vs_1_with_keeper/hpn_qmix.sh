#!/bin/bash

for _ in {1..3}; do
    CUDA_VISIBLE_DEVICES="5" python ../../main.py --config=hpn_qmix --env-config=academy_3_vs_1_with_keeper with use_wandb=True group_name=hpn_qmix;
done