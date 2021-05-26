#!/bin/bash

cd $(dirname $0)

# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/cifar-10/'
runtime=$(date "+%m%d-%H%M")

echo "rerun nni params"
python main.py --dataset $dataset --model densenet121 --initial_lr 0.0013946631449685872 --weight_decay 0.0000032900507579045906 --cutout 8 --batch_size 256 