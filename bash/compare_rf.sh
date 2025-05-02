#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/compare_real_fake_patches.py \
  --real_patch_path ./data_processed_patch/patches/DrugName/Group/M0001/measurement_patch_10.npy \
  --fake_patch_path ./data_fake_augmented/patches/DrugName/Group/M0001/measurement_patch_10.npy \
  --output_dir ./patch_comparisons \
  --pixels 112,112 50,50
