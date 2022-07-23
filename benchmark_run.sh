#!/bin/bash

PASSTHRU_ARGS="${@}"


SHORT=i:,t:,n:,e:,s,h
LONG=instanceList:,resultTag:,numWorkerProcesses:,epochTime:,static,help
OPTS=$(getopt -q --alternative --name benchmark_run --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

# extract options and their arguments into variables.
staticFlag=""

function usage {
        echo "./$(basename $0) -i instanceList -t resultTag [-n numWorkerProcesses] [-s] [tuning parameters passed to solver.py]"
        echo "-i: used to specify list of instances (required). This should be a newline-separated text file with one instance filename per line."
        echo "-t: used to specify a tag (required). The result files will be placed in a subdirectory with this name."
        echo "-n: used to specify the number of concurrent processes (default=6)."
        echo "-e: used to specify the maximum seconds per epoch (default=5)."
        echo "-s: used to specify static run. Omit to run dynamic."
        echo "-h: used to display the help menu"
}

# Global Parameters
unset -v instanceList
unset -v resultTag
numWorkerProcesses=6
staticFlag=""

# Controller Parameters
epochTime=5 # time limit for one epoch 

while true ; do
case "$1" in
-i|--instanceList)
instanceList=$2 ; shift 2 ;;
-t|--resultTag)
resultTag=$2 ; shift 2 ;;
-n|--numWorkerProcesses)
numWorkerProcesses=$2 ; shift 2 ;;
-e|--epochTime)
epochTime=$2 ; shift 2 ;;
-s|--static)
staticFlag="--static" ; shift ;;
-h|--help)
usage ; exit ;;
\?) shift ; break ;;
*) shift ; break ;;
esac
done

if [ -z "$instanceList" ]; then
        echo 'Missing -i' >&2
        exit 1
fi
if [ -z "$resultTag" ]; then
        echo 'Missing -t' >&2
        exit 1
fi

# Global Parameters (hardcoded)
instanceDir=instances/ # relative path to instance data files
resultDir=results/ # relative parent directory to save results files (will create a subdir based on the local git branch name)

# Solver-specific Parameters
strategy=greedy
verboseFlag=""

# Read the list of instances
instances=($(cat $instanceList))
num_instances=${#instances[@]}
echo "Number of instances: ${num_instances}"

# Get name of current branch
resultSubDir=${resultTag}/

# Create a directory for output
mkdir -p $resultDir$resultSubDir

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
            fname="${resultDir}${resultSubDir}${instances[$idx]}"
            python controller.py --instance ${instanceDir}${instances[$idx]} --epoch_tlim $epochTime ${staticFlag} -- python solver.py --strategy ${strategy} ${verboseFlag} $PASSTHRU_ARGS > $fname & 
        fi
    done
    wait
done

# Read timer
RUNTIME=`date +%T -d "1/1 + $SECONDS sec"`
echo "Completed in $RUNTIME"