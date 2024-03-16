use_par="par"

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
    ~/.julia/bin/mpiexecjl -n 120 julia test/test.jl 3 CF+MILP+SG 1 par "data"
else
    echo "serial"
    julia test/test.jl 2 CF+MILP+SG 1 sl "data"
fi
