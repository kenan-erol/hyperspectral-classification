#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/augment_patches.py   --input_dir './data_processed_patch/patches/'   --output_dir './data_real_fake/fake'   --noise_std_dev 0.25  --apply_scaling --scale_factor_range 0.6 1.4 --apply_offset --offset_range -0.2 0.2 --seed 42  --visualize_count 9   --visualize_dir ./data_fake/visualizations --input_label_filename './data_processed_patch/labels_patches.txt'
