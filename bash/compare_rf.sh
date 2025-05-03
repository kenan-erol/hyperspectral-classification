#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/compare_real_fake_patches.py \
  --real_patch_path 'data_real_fake/real/Tramadol 2025-01-14/Tramadol 2025-01-14/M0001/measurement_patch_8.npy' \
  --fake_patch_path './data_real_fake/fake/patches_augmented/Tramadol 2025-01-14/Tramadol 2025-01-14/M0001/measurement_patch_8.npy' \
  --output_dir ./patch_comparisons \
  --pixels 112,112 0,0
