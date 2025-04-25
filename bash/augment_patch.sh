#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/augment_patches.py \
    --input_dir ./data_processed_patch/ \
    --output_dir ./data_augmented_noise_0.1_subset/ \
    --noise_std_dev 0.1 \
    --max_patches 1000