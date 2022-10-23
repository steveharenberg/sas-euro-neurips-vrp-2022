#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --pruneRoutes --strategy fdist --thresholdSchedule 0.65,1.0,0.8,0.8,0.55,0.15,0.3,0.2 --randomGenerator 3 $PASSTHRU_ARGS