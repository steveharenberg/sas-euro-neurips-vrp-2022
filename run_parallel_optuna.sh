#!/bin/bash
nw=$1  # parallelism degree

instances=( $(ls -1 ./instances/*) )
num_instances=${#instances[@]}

for i in $(seq 1 $nw)
do
   python tune_thresholds.py &
done
wait
