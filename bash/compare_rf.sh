#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/compare_real_fake_patches.py \
  --real_patch_path 'data_real_fake/real/Klonazepam 2024-12-09/Group/M0003/measurement_patch_80.npy' \
  --fake_patch_path 'data_real_fake/fake/patches_augmented/Klonazepam 2024-12-09/Group/M0003/measurement_patch_80.npy' \
  --output_dir ./patch_comparisons \
  --pixels 112,112 0,0
