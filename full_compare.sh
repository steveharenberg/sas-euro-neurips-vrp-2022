#!/bin/bash

nw=7
instanceList=instances_249.txt
epochTime=0

EXTRA_ARGS="-t full_baselineT2"
# EXTRA_ARGS="-t tuning_manual1 --preprocessTimeWindows 1"
for SOLVER_SEED in $1 # 2 3 4 5
   do
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
done