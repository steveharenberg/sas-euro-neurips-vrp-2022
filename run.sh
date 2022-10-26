#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --pruneRoutes --strategy fdist_vroom --thresholdSchedule 1.000,0.9000,0.8100,0.7290,0.6561,0.5905,0.5314,0.4783 --randomGenerator 3 --exploreLevel 3 $PASSTHRU_ARGS
