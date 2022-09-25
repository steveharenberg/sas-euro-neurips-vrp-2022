#!/bin/bash

instanceSize=S
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

  *)
    echo "Bad instanceSize option."
    exit
    ;;
esac

epochTime=0

PREFIX="quick${instanceSize}"
DYNAMICFLAG=""
case $DYNAMICFLAG in
 D)
    staticFlag=""
    ;;

  *)
    staticFlag=" -s"
    ;;
esac


for SOLVER_SEED in 1 #2 3 4 5
   do
   # EXTRA_ARGS="-t ${PREFIX}_baseline3"
   # ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
   # EXTRA_ARGS="-t ${PREFIX}_manual1 --fractionGeneratedRandomly 0"
   # ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS

   
   # EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_baseline2"
   # ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime $staticFlag -d $SOLVER_SEED $EXTRA_ARGS
   # EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_manual1 --randomGenerator 1"
   EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_manual3 --solver_seed 212165  --randomGenerator 3"
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime $staticFlag -d $SOLVER_SEED $EXTRA_ARGS
done
echo "Finished. Tabulating results..."

# bash tabulate_results.sh ${PREFIX} > "results/quick-results-${instanceSize}.tsv"
bash tabulate_results.sh ${DYNAMICFLAG}${PREFIX} > "results/${DYNAMICFLAG}quick-results-${instanceSize}.tsv"

