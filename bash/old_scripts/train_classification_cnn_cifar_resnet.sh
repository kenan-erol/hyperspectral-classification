#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_cnn.py \
--n_batch 64 \
--dataset 'cifar10' \
--encoder_type 'resnet18' \
--n_epoch 200 \
--learning_rate 0.09 \
--learning_rate_decay 0.6 \
--learning_rate_period 25 \
--checkpoint_path 'cifar_checkpoints/cifar_cnn_checkpoint_resnet.pth' \
--device 'cuda' \
