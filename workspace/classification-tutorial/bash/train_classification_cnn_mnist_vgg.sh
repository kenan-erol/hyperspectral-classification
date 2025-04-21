#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_classification_cnn.py \
--n_batch 64 \
--dataset 'mnist' \
--encoder_type 'vggnet11' \
--n_epoch 100 \
--learning_rate 0.05 \
--learning_rate_decay 0.3 \
--learning_rate_period 40 \
--checkpoint_path 'mnist_checkpoints/mnist_cnn_checkpoint_vgg.pth' \
--device 'cuda' \
