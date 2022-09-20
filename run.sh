#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 1909 --strategy rdist --randomGenerator 1 $PASSTHRU_ARGS