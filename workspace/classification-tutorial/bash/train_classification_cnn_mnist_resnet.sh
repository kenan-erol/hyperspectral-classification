#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_cnn.py \
--n_batch 64 \
--dataset 'mnist' \
--encoder_type 'resnet18' \
--n_epoch 50 \
--learning_rate 0.1 \
--learning_rate_decay 0.1 \
--learning_rate_period 25 \
--checkpoint_path 'mnist_checkpoints/mnist_cnn_checkpoint_resnet.pth' \
--device 'cuda' \
