#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_hyper.py \
  --n_batch 25 \
  --data_dir './data_real_fake' \
  --label_file './labels_real_fake.txt' \
  --patch_size 224 \
  --train_split_ratio 0.8 \
  --encoder_type 'resnet18' \
  --num_channels 256 \
  --checkpoint_path 'hyper_checkpoints/resnet_rf/model-36.pth' \
  --device 'cuda'
