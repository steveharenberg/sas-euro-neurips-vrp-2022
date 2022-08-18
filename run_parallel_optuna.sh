#!/bin/bash
nw=$1  # parallelism degree

instances=( $(ls -1 ./instances/*) )
num_instances=${#instances[@]}

for i in $(seq 0 $nw)
do
   python tune_staged.py &
done
wait
