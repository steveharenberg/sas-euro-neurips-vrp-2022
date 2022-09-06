#!/bin/bash

nw=8
instanceList=instances_8_small.txt
epochTime=0



# EXTRA_ARGS="-t quick_baseline"
EXTRA_ARGS="-t quick_manual1 --preprocessTimeWindows 1"
for SOLVER_SEED in 1 2 3 4 5
   do
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
done
echo "Finished. Tabulating results..."

bash "tabulate_results.sh" quick_ > results/quick-results.tsv