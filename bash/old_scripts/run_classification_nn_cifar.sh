#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_nn.py \
--n_batch 25 \
--dataset 'cifar10' \
--checkpoint_path 'cifar_checkpoints/cifar_nn_checkpoint.pth' \
--output_path 'outputs/cifar_nn_output.png' \
--device 'gpu' \
