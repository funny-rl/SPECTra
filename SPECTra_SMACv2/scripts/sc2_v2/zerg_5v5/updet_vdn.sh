#!/bin/bash

for _ in {1..3}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=updet_vdn --env-config=sc2_v2_zerg with \
    use_wandb=True group_name=updet_vdn;
done