# run benchmark_run.sh for the strategy saved in the branch given by first parameter

tempBranch=`git rev-parse --abbrev-ref HEAD`
git checkout --quiet $1
for SOLVER_SEED in 1 2 3 4 5
do
   bash ./benchmark_run.sh -i instances_25.txt -t $1 --static -n 25 -d $SOLVER_SEED
done
git checkout --quiet $tempBranch