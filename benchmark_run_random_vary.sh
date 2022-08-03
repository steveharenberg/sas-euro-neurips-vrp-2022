# run benchmark_run.sh for the strategy saved in the branch given by first parameter

tempBranch=`git rev-parse --abbrev-ref HEAD`
BRANCH=strategy_random
# BRANCH=strategy_vroom_random

git stash
git checkout --quiet $BRANCH
for p in {0..100..10}
do
    c=$(echo "$p*0.01"|bc)
    bash ./benchmark_run.sh -i instances_25.txt -t D${BRANCH}_$c -e 60 -n 5 --randomprob $c
done
git checkout --quiet $tempBranch
git stash pop