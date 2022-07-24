#!/bin/bash

NUMACTL="numactl --cpunodebind=1 --preferred=1"

for TIMELIM in 30 60 120
do
    OUTDIR="results/static/baseline_$TIMELIM"
    mkdir -p $OUTDIR
    for f in instances/*
    do
        echo "$TIMELIM, $f"
        #sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
        $NUMACTL python controller.py --instance $f --epoch_tlim $TIMELIM --static -- python solver.py --verbose > $OUTDIR/$(basename $f)
    done
done