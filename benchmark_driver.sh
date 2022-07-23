# run benchmark_run.sh in each of several strategies (saved in unique branches called strategy*)

tempBranch=`git rev-parse --abbrev-ref HEAD`

eval $(git for-each-ref --shell --format='echo %(refname); git checkout --quiet %(refname); bash ./benchmark_run.sh -i instances_10.txt -t $(basename %(refname)) -n 6;' refs/heads/strategy*)

git checkout $tempBranch