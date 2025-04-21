#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/train_classification_cnn.py \ TODO: change this
--n_batch 64 \
--dataset 'cifar10' \ TODO: change this
--encoder_type 'resnet18' \
--n_epoch 200 \
--learning_rate 0.09 \
--learning_rate_decay 0.6 \
--learning_rate_period 25 \
--checkpoint_path 'hyper_checkpoints/hyper_cnn_checkpoint_resnet.pth' \
--device 'cuda' \
