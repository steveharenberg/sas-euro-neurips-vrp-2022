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
DYNAMICFLAG="D"
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
  #  EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_routepruning1 --solver_seed 212165  --randomGenerator 3 --strategy fdist"
  #  EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_schedule1 --solver_seed 212165  --randomGenerator 3 --strategy fdist --thresholdSchedule 0.8,0.7,0.6,0.5,0.4,0.3"
  #  EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_schedule2 --solver_seed 212165  --randomGenerator 3 --strategy fdist --thresholdSchedule 0.8,0.95,0.8,0.8,0.5,0.25,0.3,0.9"
   EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_schedule2 --solver_seed 212165  --randomGenerator 3 --strategy fdist --thresholdSchedule 0.85,0.7,0.8,0.75,0.55,0.15,0.3,0.3"
  #  EXTRA_ARGS="-t ${DYNAMICFLAG}${PREFIX}_schedule1 --solver_seed 212165  --randomGenerator 3 --strategy fdist --thresholdSchedule 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime $staticFlag -d $SOLVER_SEED $EXTRA_ARGS
done
echo "Finished. Tabulating results..."

# bash tabulate_results.sh ${PREFIX} > "results/quick-results-${instanceSize}.tsv"
bash tabulate_results.sh ${DYNAMICFLAG}${PREFIX} > "results/${DYNAMICFLAG}quick-results-${instanceSize}.tsv"

