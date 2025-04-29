#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/prepare_real_fake_labels.py --data_dir ./data_real_fake --output ./labels_real_fake.txt