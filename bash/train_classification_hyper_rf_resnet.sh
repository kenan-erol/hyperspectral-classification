#!/bin/bash

# pwd
# cd $SLURM_SUBMIT_DIR
# pwd

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_hyper.py \
--n_batch 25 \
--data_dir './data_processed_patch/patches' \
--label_file './data_processed_patch/labels_patches.txt' \
--patch_size 224 \
--train_split_ratio 0.8 \
--encoder_type 'resnet18' \
--num_channels 256 \
--n_epoch 50 \
--learning_rate 0.001 \
--learning_rate_decay 0.5 \
--learning_rate_period 10 \
--checkpoint_path 'hyper_checkpoints/resnet_rf/' \
--device 'cuda'
