#!/bin/bash

# Get default values of command line arguments
qcfile="qc"     # The "root" of the Q-Chem input file the string prior to ".in" and ".out"
nsurf=1         # How many surfaces to consider in the dynamics
ncen=0          # Total number of basis function centers in calcualtion (number of atoms unless ghost centers are used)

# Check for command-line arguments and override the default values
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qcfile)
            qcfile="$2"
            shift 2
            ;;
        --nsurf)
            nsurf="$2"
            shift 2
            ;;
        --ncen)
            ncen="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if optimization successfully converged 
if ! grep -q "OPTIMIZATION CONVERGED" "${qcfile}.out"; then
    echo "Error: optimization did not converge in the previous step, terminating run."
    exit 1
fi

# Extract the converged molecular geometry, store in xfile.txt
grep -A$((ncen+4)) "OPTIMIZATION CONVERGED" ${qcfile}.out | sed '1,5d' | sed 's/^[[:space:]]*[0-9]\+[[:space:]]\+//' > xfile.txt

# Extract the energies
for ((i = 0; i < nsurf; i++)); do
    awk -v pattern="E_${i}" '$0 ~ "^ " pattern " = " {last_match = $0} END {split(last_match, result, " = "); print result[2]}' "${qcfile}.out" > "E${i}.txt"
done

# Extract gradient vectors 
for ((i = 0; i < nsurf; i++)); do
    grep -A "$ncen" "Gradient of E${i}:" "${qcfile}.out" | tail -"$ncen" > "delE${i}.txt"
done

# Extract derivative couplings
for ((i = 0; i < nsurf; i++)); do
    for ((j = 0; j < i; j++)); do
        grep -A "$ncen" "Derivative Coupling d${i}${j}:" "${qcfile}.out" | tail -"$ncen" > "d${i}${j}.txt"
    done
done