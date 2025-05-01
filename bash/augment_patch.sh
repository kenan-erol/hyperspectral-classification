#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/augment_patches.py   --input_dir './data_processed_patch/patches/'   --output_dir './data_real_fake/fake'   --noise_std_dev 0.5   --visualize_count 5   --visualize_dir ./data_augmented_noise_0.5/visualizations --input_label_filename './data_processed_patch/labels_patches.txt'
