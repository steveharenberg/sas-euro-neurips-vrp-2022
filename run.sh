#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --strategy rdist --randomGenerator 3 $PASSTHRU_ARGS