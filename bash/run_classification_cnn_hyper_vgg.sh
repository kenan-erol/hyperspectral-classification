#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_classification_cnn.py \ TODO: change this
--n_batch 25 \
--dataset 'cifar10' \ TODO: change this
--encoder_type 'vggnet11' \
--checkpoint_path 'hyper_checkpoints/hyper_cnn_checkpoint_vgg.pth/model-99.pth' \
--output_path 'outputs/hyper_cnn_output_vgg.png' \
--device 'cuda' \
