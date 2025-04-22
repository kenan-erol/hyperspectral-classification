#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/preprocess_a2s2k.py \
    --data_dir ./data_processed \
    --label_file ./labels.txt \
    --output_dir ./data_a2s2k \
    --sam2_checkpoint_path ./sam2/checkpoints/sam2.1_hiera_base_plus.pt \
    --patch_size 224 \
    --num_patches_per_image 5 \
    --device cuda