#!/bin/bash
nw=$1  # parallelism degree

instances=( $(ls -1 ./instances/*) )
num_instances=${#instances[@]}

for i in $(seq 0 $nw)
do
   python hidden/tune_static.py &
done
wait
