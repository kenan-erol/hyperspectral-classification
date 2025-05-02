#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/augment_patches.py   --input_dir './data_processed_patch/patches/'   --output_dir './data_fake'   --noise_std_dev 0.2  --apply_scaling --scale_factor_range 0.8 1.2 --apply_offset --offset_range -0.02 0.02 --seed 42  --visualize_count 10   --visualize_dir ./data_augmented_noise_0.5/visualizations --input_label_filename './data_processed_patch/labels_patches.txt'
