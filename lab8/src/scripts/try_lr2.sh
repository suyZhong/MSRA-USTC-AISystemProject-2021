#!/bin/bash

cd $(dirname $0)

# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/cifar-10/'
runtime=$(date "+%m%d-%H%M")

# echo 'part 1: default parameters'
# python main.py --dataset $dataset 

echo 'part 2: simply try the lr'
for lr in 0.15 0.2 0.5 
do
    echo "try lr=$lr"
    python main.py --dataset $dataset --initial_lr $lr
done