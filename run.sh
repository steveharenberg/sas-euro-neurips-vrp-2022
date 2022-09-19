#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --solver_seed 1309 --strategy rdist $PASSTHRU_ARGS