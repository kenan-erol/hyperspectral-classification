#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_classification_cnn.py \
--n_batch 25 \
--dataset 'cifar10' \
--encoder_type 'vggnet11' \
--checkpoint_path 'cifar_checkpoints/cifar_cnn_checkpoint_vgg.pth/model-99.pth' \
--output_path 'outputs/cifar_cnn_output_vgg.png' \
--device 'cuda' \
