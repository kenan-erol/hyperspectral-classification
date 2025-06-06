#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_hyper.py \
  --n_batch 128 \
  --data_dir './data_real_fake' \
  --label_file './labels_real_fake3.txt' \
  --patch_size 224 \
  --encoder_type 'resnet18' \
  --num_channels 256 \
  --checkpoint_path 'hyper_checkpoints/resnet_rf/model-33.pth' \
  --device 'cuda'
