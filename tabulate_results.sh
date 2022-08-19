if [[ $# -lt 1 ]];then
   prefix=strategy_
#    echo "Default prefix: $prefix"
else
   prefix=$1 # Get name of current branch. Used as the subdir of the results directory
#    echo "From Argument 1: prefix: $prefix"
fi

resultDir=results/

strategies=($(ls -1 $resultDir |grep ^$prefix))
numStrategies=${#strategies[@]}

for i in $(seq 0 $(($numStrategies - 1))); do
    instances=($(ls -1 $resultDir${strategies[$i]}))
    numInstances=${#instances[@]}

    for j in $(seq 0 $(($numInstances - 1))); do
        # extract the cost from the second line of results file
        cost=$(head $resultDir${strategies[$i]}/${instances[$j]} -n 2 | tail -n 1 | awk '{print $4}')

        if [[ $# -lt 2 ]]; then
            dest=""
            printf "${strategies[$i]}\t${instances[$j]}\t${cost}\n"
        else
            dest=$2 # table output destination
            printf "${strategies[$i]}\t${instances[$j]}\t${cost}\n" > $dest
        fi
    done
done
