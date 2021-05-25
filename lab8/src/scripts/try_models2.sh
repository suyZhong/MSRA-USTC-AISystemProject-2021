#!/bin/bash

cd $(dirname $0)

# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/cifar-10/'
runtime=$(date "+%m%d-%H%M")

# echo 'part 1: default parameters'
# python main.py --dataset $dataset 

echo 'part 3: try different models'
for model in 'densenet121' 'squeezenet1_1' 'shufflenet_v2_x1_0' 'mobilenet_v2'
do
    echo "try model $model"
    python main.py --dataset $dataset --model $model
done