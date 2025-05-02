#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_hyper.py \
  --n_batch 25 \
  --data_dir './data_processed_patch/patches' \
  --label_file './data_processed_patch/labels_patches.txt' \
  --patch_size 224 \
  --encoder_type 'resnet18' \
  --num_channels 256 \
  --checkpoint_path 'hyper_checkpoints/resnet/model-24.pth' \
  --device 'cuda'
