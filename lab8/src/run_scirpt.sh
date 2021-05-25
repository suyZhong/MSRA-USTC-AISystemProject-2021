#!/bin/bash
cd $(dirname $0)

logdir="/output/"

cp scripts/$1 .
/bin/bash $1 1>$logdir"log.txt" 2>$logdir"err.txt"
rm -f $1
