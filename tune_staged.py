import optuna
import argparse
import subprocess
import logging
import sys
import os
import uuid
import platform
import numpy as np

import tools
from environment import VRPEnvironment, ControllerEnvironment
from solver import run_baseline
from baselines.strategies import STRATEGIES

SEEDS = [1,2,3,4,5]
INSTANCE_LIST_FILENAME_25 = 'instances_25.txt'
INSTANCE_LIST_FILENAME_249 = 'instances_249.txt'
INSTANCE_LIST_FILENAME_250 = 'instances_249.txt'

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def objective(trial):
    args = AttrDict()
    args["solver_seed"] = 1234
    args["verbose"] = False
    args["epoch_tlim"] = 10
    args["strategy"] = "greedy"
    
    args["fractionGeneratedNearest"] = trial.suggest_float("fractionGeneratedNearest", 0.0, 0.3)
    args["fractionGeneratedFurthest"] = trial.suggest_float("fractionGeneratedFurthest", 0.0, 0.3)
    args["fractionGeneratedSweep"] = trial.suggest_float("fractionGeneratedSweep", 0.0, 0.3)
    args["fractionGeneratedRandomly"] = 1.0 - args["fractionGeneratedNearest"] - args["fractionGeneratedFurthest"] - args["fractionGeneratedSweep"]
    args["nbGranular"] = trial.suggest_int("nbGranular", 2, 99)
    args["minSweepFillPercentage"] = trial.suggest_int("minSweepFillPercentage", 30, 99)
    args["maxToleratedCapacityViolation"] = trial.suggest_int("maxToleratedCapacityViolation", 1, 99)
    args["maxToleratedTimeWarp"] = trial.suggest_int("maxToleratedTimeWarp", 50, 200)
    args["initialTimeWarpPenalty"] = trial.suggest_float("initialTimeWarpPenalty", 0.0, 4.0)
    args["penaltyBooster"] = trial.suggest_float("penaltyBooster", 0.0, 4.0)
    args["minimumPopulationSize"] = trial.suggest_int("minimumPopulationSize", 10, 100)
    args["generationSize"] = trial.suggest_int("generationSize", 10, 100)
    args["nbElite"] = trial.suggest_int("nbElite", 2, 20)
    args["nbClose"] = trial.suggest_int("nbClose", 2, 20)
    args["targetFeasible"] = trial.suggest_float("targetFeasible", 0.05, 1.0)
    args["repairProbability"] = trial.suggest_int("repairProbability", 1, 99)
    args["growNbGranularAfterNonImprovementIterations"] = trial.suggest_int("growNbGranularAfterNonImprovementIterations", 0, 20)
    args["growNbGranularAfterIterations"] = trial.suggest_int("growNbGranularAfterIterations", 0, 20)
    args["growNbGranularSize"] = trial.suggest_int("growNbGranularSize", 1, 20)
    args["growPopulationAfterNonImprovementIterations"] = trial.suggest_int("growPopulationAfterNonImprovementIterations", 0, 20)
    args["growPopulationAfterIterations"] = trial.suggest_int("growPopulationAfterIterations", 0, 20)
    args["growPopulationSize"] = trial.suggest_int("growPopulationSize", 1, 20)
    args["intensificationProbabilityLS"] = trial.suggest_int("intensificationProbabilityLS", 1, 99)
    args["diversityWeight"] = trial.suggest_float("diversityWeight", 0.05, 1.0)
    args["useSwapStarTW"] = trial.suggest_int("useSwapStarTW", 0, 1)
    args["skipSwapStarDist"] = trial.suggest_int("skipSwapStarDist", 0, 1)
    args["circleSectorOverlapToleranceDegrees"] = trial.suggest_int("circleSectorOverlapToleranceDegrees", 2, 90)
    args["minCircleSectorSizeDegrees"] = trial.suggest_int("minCircleSectorSizeDegrees", 2, 90)
    args["preprocessTimeWindows"] = trial.suggest_int("useSwapStarTW", 0, 1)
    print(args)
    avg_reward = 0
    instance_seed = 1
    with open(INSTANCE_LIST_FILENAME_25, 'r') as f:
        instances_25 = f.readlines()
    with open(INSTANCE_LIST_FILENAME_249, 'r') as f:
        instances_249 = f.readlines()
    k = 0
    step = 0
    for instance in instances_249:
         if k > 0 and k % 5 == 0:
            trial.report(-avg_reward, step)
            step = step + 1
            if trial.should_prune():
               raise optuna.TrialPruned()

         print(f"starting instance {k+1} of {len(instances_249)} <{instance.strip()}>")
         instance = f"instances/{instance.strip()}"
         
         # Generate random tmp directory
         tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
         args.tmp_dir = tmp_dir
         cleanup_tmp_dir = True
            
         try:
            assert instance is not None, "Please provide an instance."
            for solver_seed in SEEDS:
               args["solver_seed"] = solver_seed
               env = VRPEnvironment(seed=instance_seed, instance=tools.read_vrplib(instance), epoch_tlim=args.epoch_tlim, is_static=True)
               reward = run_baseline(args, env)
               print(f"Seed {solver_seed}, Cost = {-reward}")
               avg_reward += reward
               sys.stdout.flush()
         except Exception as e:
            print(e)
            _BIG_NUMBER = 1e9
            print(f"Cost = {_BIG_NUMBER}")
            sys.stdout.flush()
            avg_reward -= _BIG_NUMBER
         finally:
            k = k + 1
            print("CLEANUP")
            if cleanup_tmp_dir:
               print(f"cleaning up ... {tmp_dir}")
               tools.cleanup_tmp_dir(tmp_dir)

   
    avg_reward /= len(instances_249)*len(SEEDS)
    print(f"Average Cost = {-avg_reward}")
    sys.stdout.flush()
    sys.stderr.flush()
    return -avg_reward




if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "postgresql://localhost:5432/template1"
    study_name = "hgs_static_staged"
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=1000)
    print(study.best_trial)