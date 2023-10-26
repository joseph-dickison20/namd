#!/bin/bash
#SBATCH --job-name=namd
#SBATCH --output=namd.out
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH --export=ALL 
netid=jad279

# USER: DEFINE DYNAMICS PARAMATERS
nsteps=20             # Number of steps to take in trajectory
nsurf=1               # How many surfaces to consider in the dynamics
dt=0.06               # Time step (in femtoseconds)
temperature=100       # Temperature (in Kelvin) used for intializing velocitites
ncen=4                # Total number of basis function centers in calcualtion (number of atoms unless ghost centers are used)
quant_centers="2,3"   # String of nuclear indices that are quantized, place spaces in between (start count at 0)
qcfile="qc"           # The "root" of the Q-Chem input file: the string prior to ".in" and ".out"
# --conv2bohr flag of main.py controls whether or not to convert Cartesian coordinates from Angstrom to bohr, see below (include flag if true, exclude flag if false)

echo -e "\n****************** LOAD MODULES, SOURCE Q-CHEM, & ACTIVATE PYTHON ****************** \n"

# Load Q-Chem and Python modules
module load GCC/10.2.0
module load CMake/3.16.4-GCCcore-10.2.0
module load OpenBLAS/0.3.12-GCC-10.2.0
module load OpenMPI/4.0.5-GCC-10.2.0
module load HDF5/1.10.1-GCCcore-6.4.0-serial
module load GCC/10.2.0
module load miniconda
module load Python/3.8.6-GCCcore-10.2.0

# Activate Python environment and source Q-Chem
conda init bash # change shell if needed
conda activate myenv # change environment if needed
source /gpfs/gibbs/pi/hammes_schiffer/jad279/.qcsetup # change path to .qcsetup if needed

echo -e "\n****************** READY TO RUN TRAJECTORY ****************** "

for ((i = 0; i <= $nsteps; i++)); do

    echo -e "\n******************"
    echo "STEP ${i}"
    echo -e "******************\n"

    # Run Q-Chem calculation and extract relevant info from output
    echo "Running Q-Chem calculation..."
    qchem -nt $SLURM_CPUS_PER_TASK ${qcfile}.in > ${qcfile}.out
    echo -e "Done\n"
    echo "Updating nuclear coordinates for next step using NAMD module..."

    ./extract.sh \
        --qcfile $qcfile \
        --nsurf $nsurf \
        --ncen $ncen

    # Run python script to get new coordinates
    python main.py \
        --nsurf $nsurf \
        --dt $dt \
        --stepnum $i \
        --temperature $temperature \
        --quant_centers $quant_centers \
        --conv2bohr # comment out this line if no conversion to bohr is needed
    
    # Create new input file 
    head -n 2 ${qcfile}.in > temp.in; cat xfile.txt >> temp.in; echo "" >> temp.in; tail -n +$((ncen+3)) ${qcfile}.in >> temp.in; mv temp.in ${qcfile}.in

    # ****************** SAVE ANY INFO YOU WANT FROM THE CURRENT RUN BELOW ******************

    # Save proton densities at all steps divisible by 5
    if [ $((i % 5)) -eq 0 ]; then
        if [ "$i" -eq 0 ]; then
            mkdir denplt
        fi
        mv pden_s0.cube denplt/${i}.cube
    fi

done

# Remove unnecessary files
./cleanup.sh \
    --qcfile $qcfile \
    --nsurf $nsurf \
    --ncen $ncen

conda deactivate # Deactivate python 
