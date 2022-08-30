#!/bin/bash
instanceDir=instances/ # relative path to instance data files
resultDir=results/cost_vs_time_O3/ # relative parent directory to save results files (will create a subdir based on the local git branch name)
epochTime=120
instanceList=instances_25.txt

# Read the list of instances
instances=($(cat $instanceList))
num_instances=${#instances[@]}
echo "Number of instances: ${num_instances}"

let "upper = $num_instances - 1"
for i in $(seq 0 $upper); do
   let "idx = $i + 1"
   # Read timer
   RUNTIME=`date +%T -d "1/1 + $SECONDS sec"`
   echo "At $RUNTIME, beginning instance $idx of $num_instances..."
   for j in $(seq 1 5); do
      echo "SEED=$j"
      # echo ${resultDir}${instances[$i]}_s${j}.txt
      python controller.py --instance ${instanceDir}${instances[$i]} --epoch_tlim $epochTime --static -- run.sh --logTimeCost --solver_seed $j 2>${resultDir}${instances[$i]}_s${j}.txt &
      python controller.py --instance ${instanceDir}${instances[$i]} --epoch_tlim $epochTime --static -- run.sh --logTimeCost --solver_seed 1 2>${resultDir}${instances[$i]}_s${j}_.txt
   done
done