#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 2909 --strategy fdist --randomGenerator 3 $PASSTHRU_ARGS