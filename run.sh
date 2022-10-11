#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 212165 --solver_seed_dynamic 2909 --strategy fdist --randomGenerator 3 $PASSTHRU_ARGS