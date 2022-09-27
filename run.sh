#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --pruneRoutes --strategy fdist --randomGenerator 3 $PASSTHRU_ARGS