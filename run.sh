#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --strategy rdist_vroom --exploreLevel 2 --randomGenerator 3 $PASSTHRU_ARGS