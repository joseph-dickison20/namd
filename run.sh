#!/bin/bash
#SBATCH --job-name=pyqctest
#SBATCH --output=pyqctest.out
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --export=ALL 
netid=jad279

echo -e "***** LOAD MODULES, SOURCE Q-CHEM, & ACTIVATE PYTHON ***** \n"

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

echo -e "\n***** READY TO RUN TRAJECTORY ***** \n"

# Run Q-Chem calculation and extract relevant info from output
echo "Running Q-Chem calculation..."
qchem -nt $SLURM_CPUS_PER_TASK qc.in > qc.out
echo "Done"
./extract.sh

# Run python script and deactivate conda envrionment once complete
echo -e "\n***** UPDATE NUCLEAR COORDINATES VIA NAMD MODULE ***** "
python main.py
conda deactivate
