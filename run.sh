#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --strategy rdist_vroom --randomGenerator 3 $PASSTHRU_ARGS