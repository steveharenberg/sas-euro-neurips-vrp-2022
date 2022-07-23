# run benchmark_run.sh for the strategy saved in the branch given by first parameter

tempBranch=`git rev-parse --abbrev-ref HEAD`
git checkout --quiet $1
bash ./benchmark_run.sh -i instances_25.txt -t $1 -e 60 --static -n 25
git checkout --quiet $tempBranch