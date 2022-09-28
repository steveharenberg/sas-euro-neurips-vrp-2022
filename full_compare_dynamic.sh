#!/bin/bash

instanceSize=250
case $instanceSize in
 S)
    instanceList=instances_8_small.txt
    nw=8
    ;;

  M)
    instanceList=instances_8_medium.txt
    nw=8
    ;;

  L)
    instanceList=instances_8_large.txt
    nw=8
    ;;

  25)
    instanceList=instances_25.txt
    nw=7
    ;;

  249)
    instanceList=instances_249.txt
    nw=8
    ;;

  250)
    instanceList=instances_250.txt
    nw=8
    ;;

  *)
    echo "Bad instanceSize option."
    exit
    ;;
esac

epochTime=0

PREFIX="full${instanceSize}"
DYNAMICFLAG="D"
case $DYNAMICFLAG in
 D)
    staticFlag=""
    ;;

  *)
    staticFlag=" -s"
    ;;
esac

SOLVER_SEED=212165
for INSTANCE_SEED in $1
do
   EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_baseline --solver_seed 212165  --randomGenerator 3 --strategy rdist"
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime $staticFlag -a $INSTANCE_SEED -d $SOLVER_SEED $EXTRA_ARGS
   # EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_subproblem1 --solver_seed 212165  --randomGenerator 3 --strategy fdist"
   # ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime $staticFlag -a $INSTANCE_SEED -d $SOLVER_SEED $EXTRA_ARGS
done
echo "Finished. Tabulating results..."

echo "Run this after all seeds complete"
echo "bash tabulate_results.sh ${DYNAMICFLAG}${PREFIX}_ > results/${DYNAMICFLAG}full-results-${instanceSize}.tsv"

