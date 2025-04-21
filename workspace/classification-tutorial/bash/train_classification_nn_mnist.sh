#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_nn.py \
--n_batch 64 \
--dataset 'mnist' \
--n_epoch 40 \
--learning_rate 0.1 \
--learning_rate_decay 0.1 \
--learning_rate_period 10 \
--checkpoint_path 'mnist_checkpoints/mnist_nn_checkpoint.pth' \
--device 'gpu' \
