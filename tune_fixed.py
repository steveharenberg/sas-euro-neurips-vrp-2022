import optuna
import argparse
import subprocess
import logging
import sys
import os
import uuid
import platform
import random
import numpy as np

import tools
from environment import VRPEnvironment, ControllerEnvironment
from solver import run_baseline
from baselines.strategies import STRATEGIES
from optuna_percentile import PercentilePruner

DO_FULL_LENGTH = False
MAX_FREE_PARAMS = 3 # can vary this many parameters at a time
SEEDS = [1,2,3]
INSTANCE_LIST_FILENAME_25 = 'instances_25.txt'
INSTANCE_LIST_FILENAME_249 = 'instances_249.txt'
default_params = {'circleSectorOverlapToleranceDegrees': 0, 'diversityWeight': 0, 'fractionGeneratedFurthest': 0.05, 'fractionGeneratedNearest': 0.05, 'fractionGeneratedSweep': 0.05, 'generationSize': 40, 'growNbGranularAfterIterations': 0, 'growNbGranularAfterNonImprovementIterations': 5000, 'growNbGranularSize': 0, 'growPopulationAfterIterations': 0, 'growPopulationAfterNonImprovementIterations': 5000, 'growPopulationSize': 0, 'initialTimeWarpPenalty': 1.0, 'intensificationProbabilityLS': 15, 'maxToleratedCapacityViolation': 50, 'maxToleratedTimeWarp': 100, 'minCircleSectorSizeDegrees': 15, 'minimumPopulationSize': 25, 'minSweepFillPercentage': 60, 'nbClose': 5, 'nbElite': 4, 'nbGranular': 32, 'penaltyBooster': 1.0, 'repairProbability': 50, 'skipSwapStarDist': 0, 'targetFeasible': 0.2, 'useSwapStarTW': 1, 'preprocessTimeWindows': 0}


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def run_instance(instance, args, solver_seed=1, instance_seed=1):   
   # Generate random tmp directory
   tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
   args.tmp_dir = tmp_dir
   cleanup_tmp_dir = True
      
   try:
      assert instance is not None, "Please provide an instance."
      args["solver_seed"] = solver_seed
      env = VRPEnvironment(seed=instance_seed, instance=tools.read_vrplib(instance), epoch_tlim=args.epoch_tlim, is_static=True)
      reward = run_baseline(args, env)
      print(f"Seed {solver_seed}, Cost = {-reward}")
      sys.stdout.flush()
   except Exception as e:
      print(e)
      _BIG_NUMBER = 1e9
      print(f"Cost = {_BIG_NUMBER}")
      sys.stdout.flush()
      reward = -_BIG_NUMBER
   finally:
      print("CLEANUP")
      if cleanup_tmp_dir:
         print(f"cleaning up ... {tmp_dir}")
         tools.cleanup_tmp_dir(tmp_dir)
   return reward

def objective(trial):
    args = AttrDict()
    args["solver_seed"] = 1234
    args["verbose"] = False
    args["epoch_tlim"] = 0
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
    args["growNbGranularAfterNonImprovementIterations"] = trial.suggest_int("growNbGranularAfterNonImprovementIterations", 0, 20000)
    args["growNbGranularAfterIterations"] = trial.suggest_int("growNbGranularAfterIterations", 0, 20)
    args["growNbGranularSize"] = trial.suggest_int("growNbGranularSize", 0, 20)
    args["growPopulationAfterNonImprovementIterations"] = trial.suggest_int("growPopulationAfterNonImprovementIterations", 0, 20000)
    args["growPopulationAfterIterations"] = trial.suggest_int("growPopulationAfterIterations", 0, 20)
    args["growPopulationSize"] = trial.suggest_int("growPopulationSize", 0, 20)
    args["intensificationProbabilityLS"] = trial.suggest_int("intensificationProbabilityLS", 1, 99)
    args["diversityWeight"] = trial.suggest_float("diversityWeight", 0.0, 1.0)
    args["useSwapStarTW"] = trial.suggest_int("useSwapStarTW", 0, 1)
    args["skipSwapStarDist"] = trial.suggest_int("skipSwapStarDist", 0, 1)
    args["circleSectorOverlapToleranceDegrees"] = trial.suggest_int("circleSectorOverlapToleranceDegrees", 0, 90)
    args["minCircleSectorSizeDegrees"] = trial.suggest_int("minCircleSectorSizeDegrees", 2, 90)
    args["preprocessTimeWindows"] = trial.suggest_int("preprocessTimeWindows", 0, 1)
    print(args)
    avg_reward = 0
    with open(INSTANCE_LIST_FILENAME_25, 'r') as f:
        instances_25 = f.readlines()
    with open(INSTANCE_LIST_FILENAME_249, 'r') as f:
        instances_249 = f.readlines()
    k = 0
    step = 0
    instances = instances_25
    for instance in instances:
         trial.report(-avg_reward, step)
         if trial.should_prune():
            raise optuna.TrialPruned()
         step = step + 1
         instance = f"instances/{instance.strip()}"
         for solver_seed in SEEDS:
            print(f"starting instance {k+1} of {len(instances)} seed={solver_seed} <{instance.strip()}>")
            reward = run_instance(instance, args, solver_seed=solver_seed)
            avg_reward += reward
         k = k + 1

    # full-length static runs, after pruning
    if DO_FULL_LENGTH:
      args["epoch_tlim"] = 0
      k = 0
      avg_reward = 0
      for instance in instances:
         trial.report(-avg_reward, step)
         if trial.should_prune():
            raise optuna.TrialPruned()
         step = step + 1
         instance = f"instances/{instance.strip()}"
         for solver_seed in SEEDS:
            print(f"starting instance {k+1} of {len(instances)} seed={solver_seed} <{instance.strip()}>")
            reward = run_instance(instance, args, solver_seed=solver_seed)
            avg_reward += reward
         k = k + 1
   
    avg_reward /= len(instances)*len(SEEDS)
    print(f"Average Cost = {-avg_reward}")
    sys.stdout.flush()
    sys.stderr.flush()
    return -avg_reward




if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "postgresql://localhost:5432/template1"
    study_name = "hgs_static_fixed_8"
    study = optuna.create_study(direction="minimize",
                                pruner=PercentilePruner(
                                 25.0, n_min_trials=4, n_warmup_steps=5, interval_steps=5
                                ),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
 
    while True:
      if len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]) > 0:
         best_params = study.best_params
         n_free_params = random.randint(1,MAX_FREE_PARAMS)
         sample_keys = random.sample(best_params.keys(), len(best_params.keys()) - n_free_params)
         fixed_params = {k: best_params[k] for k in sample_keys}
         partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, optuna.samplers.RandomSampler())
         print(f"Trying free params: {set(best_params.keys())-set(fixed_params.keys())}")
         study.sampler = partial_sampler
         study.optimize(objective, n_trials=1)
      else:
         print("No previous trial completed. Running with default params")
         fixed_params = default_params
         partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, optuna.samplers.RandomSampler())
         study.sampler = partial_sampler
         study.optimize(objective, n_trials=1)
    print(study.best_trial)