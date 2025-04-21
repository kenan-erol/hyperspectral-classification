#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_cnn.py \
--n_batch 64 \
--dataset 'cifar10' \
--encoder_type 'vggnet11' \
--n_epoch 100 \
--learning_rate 0.1 \
--learning_rate_decay 0.1 \
--learning_rate_period 35 \
--checkpoint_path 'cifar_checkpoints/cifar_cnn_checkpoint_vgg.pth' \
--device 'cuda' \
