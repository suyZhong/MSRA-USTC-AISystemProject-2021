#!/bin/bash

cd $(dirname $0)

# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/cifar-10/'
runtime=$(date "+%m%d-%H%M")

echo "rerun nni params"
python main.py --dataset $dataset --model densenet121 --initial_lr 0.05388152470872354\ 
    --weight_decay 0.00007749766814161883 --cutout 0 --batch_size 512 --epochs 300 --optimizer sgd