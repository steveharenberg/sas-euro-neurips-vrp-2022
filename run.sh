#!/bin/bash
PASSTHRU_ARGS="${@}"
python solver.py --preprocessTimeWindows 1 --strategy fangle $PASSTHRU_ARGS