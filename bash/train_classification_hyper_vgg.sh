#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_hyper.py \
--n_batch 25 \
--data_dir './data-processed/' \
--label_file './labels.txt' \
--num_patches_per_image 5 \
--patch_size 224 \
--train_split_ratio 0.8 \
--encoder_type 'vggnet11' \
--num_channels 256 \
--n_epoch 50 \
--learning_rate 0.001 \
--learning_rate_decay 0.5 \
--learning_rate_period 10 \
--checkpoint_path 'hyper_checkpoints/vgg/' \
--device 'cuda'