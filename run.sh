num=(0 1 2 3 4 5 6 7 8 9)

dir=ban10

/home/matsuda/bin/parallel --bar -j 10 "\
    ./a.out < ./${dir}/ban0{1} > ./${dir}>ans0{1}
    " ::: ${num[@]}

