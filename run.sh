#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --solver_seed_dynamic 1310 --pruneRoutes --strategy fdist --thresholdSchedule 0.85,0.7,0.8,0.75,0.55,0.45,0.3,0.3 --randomGenerator 3 $PASSTHRU_ARGS