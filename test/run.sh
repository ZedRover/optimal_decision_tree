use_par="serial"

while getopts ":p:" opt; do
    case $opt in
        p)
            use_par=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

if [ "$use_par" == "par" ]; then
    echo "parallel"
    ~/.julia/bin/mpiexec -n 120 julia test/test.jl 2 CF+MILP+SG 1 par "small_toy"
else
    echo "serial"
    julia test/test.jl 2 CF+MILP+SG 1 sl "small_toy"
fi
