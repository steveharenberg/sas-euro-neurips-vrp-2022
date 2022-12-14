#!/bin/bash

nw=7
instanceList=instances_25.txt
epochTime=0


# from optuna
# EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 37 --diversityWeight 0.529474385295042 --fractionGeneratedFurthest 0.287055946927403 --fractionGeneratedNearest 0.128873089578007 --fractionGeneratedSweep 0.176369951949128 --generationSize 84 --growNbGranularAfterIterations 7 --growNbGranularAfterNonImprovementIterations 12 --growNbGranularSize 19 --growPopulationAfterIterations 6 --growPopulationAfterNonImprovementIterations 17 --growPopulationSize 8 --initialTimeWarpPenalty 3.40376971944594 --intensificationProbabilityLS 55 --maxToleratedCapacityViolation 50 --maxToleratedTimeWarp 187 --minCircleSectorSizeDegrees 25 --minimumPopulationSize 19 --minSweepFillPercentage 97 --nbClose 15 --nbElite 8 --nbGranular 66 --penaltyBooster 1.67327576103869 --repairProbability 66 --skipSwapStarDist 0 --targetFeasible 0.206256217856227 --useSwapStarTW 1"
# EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 4 --diversityWeight 0.166236084195633 --fractionGeneratedFurthest 0.181547254350519 --fractionGeneratedNearest 0.0623893278005611 --fractionGeneratedSweep 0.0151719396982409 --generationSize 85 --growNbGranularAfterIterations 0 --growNbGranularAfterNonImprovementIterations 19 --growNbGranularSize 14 --growPopulationAfterIterations 5 --growPopulationAfterNonImprovementIterations 18 --growPopulationSize 15 --initialTimeWarpPenalty 3.36534025944167 --intensificationProbabilityLS 81 --maxToleratedCapacityViolation 8 --maxToleratedTimeWarp 173 --minCircleSectorSizeDegrees 61 --minimumPopulationSize 31 --minSweepFillPercentage 72 --nbClose 9 --nbElite 16 --nbGranular 36 --penaltyBooster 2.13991722073726 --repairProbability 95 --skipSwapStarDist 0 --targetFeasible 0.222717992499004 --useSwapStarTW 1"
# EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 4 --diversityWeight 0.163183762649742 --fractionGeneratedFurthest 0.182716164139785 --fractionGeneratedNearest 0.0905813071636389 --fractionGeneratedSweep 0.0160586684231415 --generationSize 85 --growNbGranularAfterIterations 0 --growNbGranularAfterNonImprovementIterations 18 --growNbGranularSize 14 --growPopulationAfterIterations 7 --growPopulationAfterNonImprovementIterations 19 --growPopulationSize 17 --initialTimeWarpPenalty 3.35769973808251 --intensificationProbabilityLS 84 --maxToleratedCapacityViolation 5 --maxToleratedTimeWarp 173 --minCircleSectorSizeDegrees 61 --minimumPopulationSize 12 --minSweepFillPercentage 71 --nbClose 9 --nbElite 16 --nbGranular 36 --penaltyBooster 2.03581171233344 --repairProbability 90 --skipSwapStarDist 0 --targetFeasible 0.216619991704403 --useSwapStarTW 1"
# EXTRA_ARGS="--circleSectorOverlapToleranceDegrees 5 --diversityWeight 0.140847761396352 --fractionGeneratedFurthest 0.158759198979329 --fractionGeneratedNearest 0.0914064233972187 --fractionGeneratedSweep 0.0186357311195463 --generationSize 85 --growNbGranularAfterIterations 0 --growNbGranularAfterNonImprovementIterations 20 --growNbGranularSize 14 --growPopulationAfterIterations 5 --growPopulationAfterNonImprovementIterations 18 --growPopulationSize 17 --initialTimeWarpPenalty 3.17921598516315 --intensificationProbabilityLS 85 --maxToleratedCapacityViolation 6 --maxToleratedTimeWarp 162 --minCircleSectorSizeDegrees 65 --minimumPopulationSize 10 --minSweepFillPercentage 77 --nbClose 9 --nbElite 16 --nbGranular 32 --penaltyBooster 1.96319438091081 --repairProbability 97 --skipSwapStarDist 0 --targetFeasible 0.221141499089876 --useSwapStarTW 1"
# EXTRA_ARGS="--fractionGeneratedFurthest 0.0297427769063673 --maxToleratedCapacityViolation 56 --diversityWeight 0.0549103294641657"
# EXTRA_ARGS="--skipSwapStarDist 1 --repairProbability 70 --fractionGeneratedNearest 0.145103803784702 --intensificationProbabilityLS 63 --targetFeasible 0.152474047152356 --generationSize 23 --growPopulationAfterIterations 18 --growNbGranularAfterIterations 16 --fractionGeneratedSweep 0.222938291216638 --useSwapStarTW 0 --diversityWeight 0.492125880210599 --growPopulationAfterNonImprovementIterations 9668 --penaltyBooster 1.01771036467219 --maxToleratedCapacityViolation 64 --preprocessTimeWindows 1 --growNbGranularAfterNonImprovementIterations 11304 --nbClose 19 --maxToleratedTimeWarp 149"
# EXTRA_ARGS="--targetFeasible 0.125630423362447 --minimumPopulationSize 17 --minSweepFillPercentage 57 --nbGranular 22"
# EXTRA_ARGS="--growNbGranularAfterIterations 6 --targetFeasible 0.125630423362447 --minSweepFillPercentage 57 --minimumPopulationSize 17 --nbGranular 44"
# EXTRA_ARGS="-t tuning_manual1 --minimumPopulationSize 12 --generationSize 20 --nbElite 2"
# EXTRA_ARGS="-t tuning_manual1 --minimumPopulationSize 12 --generationSize 20 --nbElite 4"
# EXTRA_ARGS="-t tuning_improved2_opt_2"
EXTRA_ARGS="-t tuning_manual1 --preprocessTimeWindows 1"
for SOLVER_SEED in 1 # 2 3 4 5
   do
   # perform baseline
   # ./benchmark_run.sh -i $instanceList -t "tuning_improved1_opt_2" -n $nw -e $epochTime -s -d $SOLVER_SEED
   # ./benchmark_run.sh -i $instanceList -t "tuning_combined" -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
   ./benchmark_run.sh -i $instanceList -n $nw -e $epochTime -s -d $SOLVER_SEED $EXTRA_ARGS
done