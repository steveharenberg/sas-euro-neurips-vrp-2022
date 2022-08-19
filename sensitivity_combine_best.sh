#!/bin/bash

nw=7
instanceList=instances_25.txt
epochTime=0


# from optuna
# EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 37 --diversityWeight 0.529474385295042 --fractionGeneratedFurthest 0.287055946927403 --fractionGeneratedNearest 0.128873089578007 --fractionGeneratedSweep 0.176369951949128 --generationSize 84 --growNbGranularAfterIterations 7 --growNbGranularAfterNonImprovementIterations 12 --growNbGranularSize 19 --growPopulationAfterIterations 6 --growPopulationAfterNonImprovementIterations 17 --growPopulationSize 8 --initialTimeWarpPenalty 3.40376971944594 --intensificationProbabilityLS 55 --maxToleratedCapacityViolation 50 --maxToleratedTimeWarp 187 --minCircleSectorSizeDegrees 25 --minimumPopulationSize 19 --minSweepFillPercentage 97 --nbClose 15 --nbElite 8 --nbGranular 66 --penaltyBooster 1.67327576103869 --repairProbability 66 --skipSwapStarDist 0 --targetFeasible 0.206256217856227 --useSwapStarTW 1"
EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 4 --diversityWeight 0.166236084195633 --fractionGeneratedFurthest 0.181547254350519 --fractionGeneratedNearest 0.0623893278005611 --fractionGeneratedSweep 0.0151719396982409 --generationSize 85 --growNbGranularAfterIterations 0 --growNbGranularAfterNonImprovementIterations 19 --growNbGranularSize 14 --growPopulationAfterIterations 5 --growPopulationAfterNonImprovementIterations 18 --growPopulationSize 15 --initialTimeWarpPenalty 3.36534025944167 --intensificationProbabilityLS 81 --maxToleratedCapacityViolation 8 --maxToleratedTimeWarp 173 --minCircleSectorSizeDegrees 61 --minimumPopulationSize 31 --minSweepFillPercentage 72 --nbClose 9 --nbElite 16 --nbGranular 36 --penaltyBooster 2.13991722073726 --repairProbability 95 --skipSwapStarDist 0 --targetFeasible 0.222717992499004 --useSwapStarTW 1"
# EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 4 --diversityWeight 0.163183762649742 --fractionGeneratedFurthest 0.182716164139785 --fractionGeneratedNearest 0.0905813071636389 --fractionGeneratedSweep 0.0160586684231415 --generationSize 85 --growNbGranularAfterIterations 0 --growNbGranularAfterNonImprovementIterations 18 --growNbGranularSize 14 --growPopulationAfterIterations 7 --growPopulationAfterNonImprovementIterations 19 --growPopulationSize 17 --initialTimeWarpPenalty 3.35769973808251 --intensificationProbabilityLS 84 --maxToleratedCapacityViolation 5 --maxToleratedTimeWarp 173 --minCircleSectorSizeDegrees 61 --minimumPopulationSize 12 --minSweepFillPercentage 71 --nbClose 9 --nbElite 16 --nbGranular 36 --penaltyBooster 2.03581171233344 --repairProbability 90 --skipSwapStarDist 0 --targetFeasible 0.216619991704403 --useSwapStarTW 1"

for SOLVER_SEED in 1 # 2 3 4 5
   do
   # perform baseline
   # ./benchmark_run.sh -i $instanceList -t "tuning_baseline" -n $nw -e $epochTime -s -d $SOLVER_SEED
   ./benchmark_run.sh -i $instanceList -t "tuning_combined" -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
done