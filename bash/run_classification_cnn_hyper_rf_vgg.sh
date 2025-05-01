#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_hyper.py \
  --n_batch 25 \
  --data_dir './data_real_fake' \
  --label_file './labels_real_fake.txt' \
  --patch_size 224 \
  --train_split_ratio 0.8 \
  --encoder_type 'vggnet11' \
  --num_channels 256 \
  --checkpoint_path 'hyper_checkpoints/vgg_rf/model-35.pth' \
  --device 'cuda'
