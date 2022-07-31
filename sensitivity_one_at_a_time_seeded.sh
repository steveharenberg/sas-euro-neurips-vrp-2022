#!/bin/bash

nw=25
instanceList=instances_25.txt
epochTime=60

# perform baseline with all default parameters

./benchmark_run.sh -i $instanceList -t "tuning_baseline" -n $nw -e $epochTime -s

# perturb parameters from defaults, one at a time

for SOLVER_SEED in 1 2 3 4 5
   for PARAM in fractionGeneratedNearest fractionGeneratedFurthest fractionGeneratedSweep
   do
      for VALUE in 0.0 0.025 0.1 0.5
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in fractionGeneratedRandomly
   do
      for VALUE in 0.25 0.5 0.75 1.0
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in nbGranular
   do
      for VALUE in 5 10 40 80
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in minSweepFillPercentage
   do
      for VALUE in 10 30 50 75
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in maxToleratedCapacityViolation
   do
      for VALUE in 10 30 60 75
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in maxToleratedTimeWarp
   do
      for VALUE in 50 75 125 200
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in initialTimeWarpPenalty penaltyBooster
   do
      for VALUE in 0.25 0.5 1.5 4.0
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in minimumPopulationSize generationSize
   do
      for VALUE in 15 35 45 60
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in nbElite nbClose
   do
      for VALUE in 2 3 6 10
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in targetFeasible
   do
      for VALUE in 0.05 0.1 0.3 0.5
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in repairProbability
   do
      for VALUE in 15 35 65 80
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in growNbGranularSize growPopulationSize
   do
      for VALUE in 2 4 8 16
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in intensificationProbabilityLS
   do
      for VALUE in 5 25 50 75 
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in diversityWeight
   do
      for VALUE in 0.1 0.2 0.5 0.9
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in useSwapStarTW
   do
      for VALUE in 0
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in skipSwapStarDist
   do
      for VALUE in 1
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done

   for PARAM in circleSectorOverlapToleranceDegrees minCircleSectorSizeDegrees
   do
      for VALUE in 5 20 40 60
      do
         ./benchmark_run.sh -i $instanceList -t "tuning_${PARAM}_${VALUE}" -n $nw -e $epochTime -s "--${PARAM}"  "$VALUE" -d $SOLVER_SEED
      done 
   done
done