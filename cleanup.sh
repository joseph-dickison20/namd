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

# Below are the cleanup commands specific for the NAMD file writes
# Can be used to cleanup directory, but I will not enable this here
:<<'COMMENT'
# Remove NAMD file writes
rm xfile.txt vfile.txt gfile.txt tdfile.txt ${qcfile}.out

# Remove energy and gradient files
for ((i = 0; i < nsurf; i++)); do
    rm E${i}.txt delE${i}.txt
done

# Remove derivative coupling files
for ((i = 0; i < nsurf; i++)); do
    for ((j = 0; j < i; j++)); do
        rm d${i}${j}.txt
    done
done
COMMENT