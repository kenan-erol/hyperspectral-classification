#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/compare_real_fake_patches.py \
  --real_patch_path 'data_processed_patch/patches/Bromazolam 2025-01-20/Group/M0001/measurement_patch_38.npy' \
  --fake_patch_path './data_fake/patches_augmented/Bromazolam 2025-01-20/Group/M0001/measurement_patch_38.npy' \
  --output_dir ./patch_comparisons \
  --pixels 112,112 50,50
