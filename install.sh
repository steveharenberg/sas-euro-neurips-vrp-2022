#!/bin/bash
pip install -r requirements.txt
cd baselines/hgs_vrptw
make clean
make all
cd ../..

cd baselines/vroom/src
make clean
make USE_ROUTING=false
cd ../..