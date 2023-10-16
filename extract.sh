#!/bin/bash

# Define input parameters
file="qc.out"   # Q-Chem output filename
nsurf=2         # How many surfaces we will consider in the dynamics
ncen=4          # Number of centers in the molecule

# Check if optimization successfully converged 
if ! grep -q "OPTIMIZATION CONVERGED" "$file"; then
    echo "Error: optimization did not converge in the previous step, terminating run."
    exit 1
fi

# Extract the converged molecular geometry
grep -A$((ncen+4)) "OPTIMIZATION CONVERGED" qc.out | sed '1,5d' | sed 's/^[[:space:]]*[0-9]\+[[:space:]]\+//' > geom.txt

# Extract the energies
for ((i = 0; i < nsurf; i++)); do
    awk -v pattern="E_${i}" '$0 ~ "^ " pattern " = " {last_match = $0} END {split(last_match, result, " = "); print result[2]}' "$file" > "E${i}.txt"
done

# Extract gradient vectors 
for ((i = 0; i < nsurf; i++)); do
    grep -A "$ncen" "Gradient of E${i}:" "$file" | tail -"$ncen" > "delE${i}.txt"
done

# Extract derivative couplings
for ((i = 0; i < nsurf; i++)); do
    for ((j = 0; j < i; j++)); do
        grep -A "$ncen" "Derivative Coupling d${i}${j}:" "$file" | tail -"$ncen" > "d${i}${j}.txt"
    done
done
