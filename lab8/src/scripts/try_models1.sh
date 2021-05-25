#!/bin/bash

cd $(dirname $0)

# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/cifar-10/'
runtime=$(date "+%m%d-%H%M")

# echo 'part 1: default parameters'
# python main.py --dataset $dataset 

echo 'part 3: try different models'
for model in 'resnet50' 'vgg16' 'vgg16_bn' 'resnext50_32x4d'
do
    echo "try model $model"
    python main.py --dataset $dataset --model $model
done