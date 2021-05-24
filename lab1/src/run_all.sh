#!/bin/bash
cd $(dirname $0)
# hyper parameters
bs=64
lr=1.0
# I run on bitahub so here is my name
dataset='/data/4ZhongSuyang/mnist_dataset'
runtime=$(date "+%m%d-%H%M")
output="/output/mnist/$runtime/"
log="/output/mnist_log/$runtime.log"

echo 'part 1: save graph'
output1=$output"graph"
python mnist_stu.py --dataset $dataset \
    --output $output1\
    --save-graph

echo 'part 2: save scalar'
output2=$output"scalar"
python mnist_stu.py --dataset $dataset \
    --output $output2\
    --save-scalar

echo 'part3: print profile'
python mnist_stu.py --dataset $dataset --profile --output $output

echo 'part4: try different bs profile (with cpu)'
for bs in 1 16 64
do
    echo "begin batchsize=$bs profiling"
    python mnist_stu.py --dataset $dataset --batch-size $bs --profile --output $output --no-cuda
done

echo 'part5: try different bs profile (with gpu)'
for bs in 1 16 64
do
    echo "begin batchsize=$bs profiling"
    python mnist_stu.py --dataset $dataset --batch-size $bs --profile --output $output
done