# run benchmark_run.sh in each of several strategies (saved in unique branches called strategy*)

tempBranch=`git rev-parse --abbrev-ref HEAD`

eval $(git for-each-ref --shell   --format='echo %(refname); git checkout --quiet %(refname); ./benchmark_run.sh instances_10.txt %(refname:lstrip=2);'   refs/heads/strategy*)

git checkout $tempBranch