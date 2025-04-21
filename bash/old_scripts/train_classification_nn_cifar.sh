#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_nn.py \
--n_batch 64 \
--dataset 'cifar10' \
--n_epoch 50 \
--learning_rate 1e-1 \
--learning_rate_decay 0.1 \
--learning_rate_period 10 \
--checkpoint_path 'cifar_checkpoints/cifar_nn_checkpoint.pth' \
--device 'gpu' \
