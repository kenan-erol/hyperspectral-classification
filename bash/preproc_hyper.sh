#!/bin/bash

# pwd
# cd $SLURM_SUBMIT_DIR
# pwd

export CUDA_VISIBLE_DEVICES=0

python tools/preproc_patch.py \
--n_batch 25 \
--data_dir './data_processed/' \
--label_file './labels.txt' \
--num_patches_per_image 100 \
--patch_size 224 \
--train_split_ratio 0.8 \
--encoder_type 'resnet18' \
--num_channels 256 \
--n_epoch 50 \
--learning_rate 0.001 \
--learning_rate_decay 0.5 \
--learning_rate_period 10 \
--checkpoint_path 'hyper_checkpoints/resnet/' \
--device 'cuda' \
--output_dir './data_processed_patch/' \
--sam2_checkpoint_path './sam2/checkpoints/sam2.1_hiera_base_plus.pt'
# --max_images 10
