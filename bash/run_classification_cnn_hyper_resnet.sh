#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_classification_cnn.py \ TODO: change this
--n_batch 25 \
--dataset 'cifar10' \ TODO: change this
--encoder_type 'resnet18' \
--checkpoint_path 'hyper_checkpoints/hyper_cnn_checkpoint_resnet.pth/model-199.pth' \
--output_path 'outputs/hyper_cnn_output_resnet.png' \
--device 'cuda' \
