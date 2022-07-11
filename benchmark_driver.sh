# run benchmark_run.sh in each of several strategies (saved in unique branches called strategy*)

tempBranch=`git rev-parse --abbrev-ref HEAD`

<<<<<<< HEAD
eval $(git for-each-ref --shell   --format='echo %(refname); git checkout --quiet %(refname); ./benchmark_run.sh instances_1.txt %(refname:lstrip=2);'   refs/heads/strategy*)
=======
eval $(git for-each-ref --shell   --format='echo %(refname); git checkout --quiet %(refname);  ./benchmark_run.sh instances_1.txt %(refname);'   refs/heads/strategy*)
>>>>>>> 40505277304ed2a3014e528e69547b5ab627fc2a

git checkout $tempBranch