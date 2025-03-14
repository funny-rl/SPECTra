#!/bin/bash


for _ in {1}; do
    CUDA_VISIBLE_DEVICES="0" python ../../../main.py --config=ss_vdn --env-config=sc2_v2_protoss with use_wandb=True group_name=spectra_vdn;
done