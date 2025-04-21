#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_nn.py \
--n_batch 25 \
--dataset 'mnist' \
--checkpoint_path 'mnist_checkpoints/mnist_nn_checkpoint.pth' \
--output_path 'outputs/mnist_nn_output.png' \
--device 'gpu' \
