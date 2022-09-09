#!/bin/bash

nw=8
# instanceList=instances_8_small.txt
# instanceList=instances_8_medium.txt
instanceList=instances_8_large.txt
# instanceList=instances_25.txt
epochTime=0

PREFIX=quickL

for SOLVER_SEED in 1 #2 3 4 5
   do
   # EXTRA_ARGS="-t ${PREFIX}_baseline1"
   # ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
   EXTRA_ARGS="-t ${PREFIX}_manual1 --fractionGeneratedRandomly 0"
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
done
echo "Finished. Tabulating results..."

bash "tabulate_results.sh" ${PREFIX} > results/quick-results-L.tsv