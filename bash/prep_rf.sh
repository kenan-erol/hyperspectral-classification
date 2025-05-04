#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python tools/prep_real_fake.py --data_dir ./data_real_fake --output ./labels_real_fake_weak.txt
