#!/bin/bash

cd $(dirname $0)

# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/cifar-10/'
runtime=$(date "+%m%d-%H%M")

echo 'part 3: try different lr on densenet'
for lr in 0.05 0.07 0.03
do
    echo "try lr $lr"
    python main.py --dataset $dataset --model densenet121 --initial_lr $lr
done