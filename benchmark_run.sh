#!/bin/bash

# TODO: implement keyword parameters, positional parameters are not very convenient

if [[ $# -eq 0 ]];then
   echo "ERROR: No arguments! usage: benchmark_run.sh instanceList [branchTag] [numWokers] [--static]"
   exit
else
   instanceList=$1 # text file containing instances separated by newline
   echo "From Argument 1: instanceList: $instanceList"
fi

if [[ $# -lt 2 ]];then
   branchTag=`git rev-parse --abbrev-ref HEAD` # Get name of current branch
   echo "Default branchTag: $branchTag"
else
   branchTag=$2 # Get name of current branch. Used as the subdir of the results directory
   echo "From Argument 2: branchTag: $branchTag"
fi

if [[ $# -lt 3 ]];then
   numWorkerProcesses=1 # parallelism degree. This many processes will spawn simultaneously
   echo "Default numWorkerProcesses: $numWorkerProcesses"
else
   numWorkerProcesses=$3 # parallelism degree. This many processes will spawn simultaneously
   echo "From Argument 3: numWorkerProcesses: $numWorkerProcesses"
fi

if [[ $# -lt 4 ]];then
   staticFlag="" # set to "--static" for static or "" for dynamic
   echo "No 4th argument. Solving dynamic."
else
   staticFlag=$4 # set to "--static" for static or "" for dynamic
   echo "From Argument 4: staticFlag: $staticFlag"
fi

# Global Parameters
instanceDir=instances/ # relative path to instance data files
resultDir=results/ # relative parent directory to save results files (will create a subdir based on the local git branch name)

# Controller Parameters
# TODO: make this a script parameter
epochTime=5 # time limit for one epoch 

# Solver-specific Parameters
strategy=greedy
verboseFlag=""

# Read the list of instances
instances=($(cat $instanceList))
num_instances=${#instances[@]}
echo "Number of instances: ${num_instances}"

# Get name of current branch
branchSubDir=${branchTag}/

# Create a directory for output
mkdir -p $resultDir$branchSubDir

# Start timer
SECONDS=0

for i in $(seq 0 $numWorkerProcesses $num_instances); do
    if [ $i -lt $num_instances ]; then
        let "idx = $i + 1"
        # Read timer
        RUNTIME=`date +%T -d "1/1 + $SECONDS sec"`
        echo "At $RUNTIME, beginning instance $idx of $num_instances..."
    fi
    for j in $(seq 0 $numWorkerProcesses); do
        let "idx = $i + $j"
        if [ $idx -lt $num_instances ]; then
            fname="${resultDir}${branchSubDir}${instances[$idx]}"
            python controller.py --instance ${instanceDir}${instances[$idx]} --epoch_tlim $epochTime ${staticFlag} -- python solver.py --strategy ${strategy} ${verboseFlag} > $fname & 
        fi
    done
    wait
done

# Read timer
RUNTIME=`date +%T -d "1/1 + $SECONDS sec"`
echo "Completed in $RUNTIME"