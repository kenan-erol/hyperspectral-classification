#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_cnn.py \
--n_batch 25 \
--dataset 'mnist' \
--encoder_type 'resnet18' \
--checkpoint_path 'mnist_checkpoints/mnist_cnn_checkpoint_resnet.pth/model-49.pth' \
--output_path 'outputs/mnist_cnn_output_resnet.png' \
--device 'cuda' \
