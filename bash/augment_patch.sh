#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/augment_patches.py   --input_dir './data_processed_patch/patches/'   --output_dir './data_real_fake/fake'   --noise_std_dev 0.05 \ 
  --apply_scaling \
  --scale_factor_range 0.9 1.1 \
  --apply_offset \
  --offset_range -0.05 0.05 \ 
  --seed 42 \
  --visualize_count 9 \   
  --visualize_dir ./data_fake/visualizations --input_label_filename './data_processed_patch/labels_patches.txt'
