import numpy as np
import argparse
from src.calc_classes import *

def main():
    
    # ******************************
    #       MAIN RUN FUNCTION
    # ******************************

    # This main.py script is the interface bewteen your ES code
    # and the dynamics code in the src/ directory, updating the 
    # position of classical nuclei via AIMD, Ehrenfest, or FSSH.

    # See the README.txt for details on the workflow. 

    # ******* READ IN & DEFINE NECESSARY PARAMETERS ******* 
    
    # Parse arguments, set defaults
    parser = argparse.ArgumentParser(description="Argument parser for NAMD module")
    parser.add_argument("--nsurf", type=int, default=1, help="How many surfaces to consider in the dynamics")
    parser.add_argument("--dt", type=float, default=0.06, help="Time step (in femtoseconds)")
    parser.add_argument("--stepnum", type=int, default=0, help="The current step number of the trajectory")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature (in Kelvin) used for intializing velocitites")
    parser.add_argument("--quant_centers", type=str, default="", help="String of nuclear indices that are quantized, place commas in between (start count at 0)")
    parser.add_argument("--conv2bohr", action='store_true', help="Controls whether or not to convert Cartesian coodfinates from Angstrom to bohr (include for true, exclude for false)")
    args = parser.parse_args()

    # Obtain values
    dt = args.dt
    nsurf = args.nsurf
    stepnum = args.stepnum
    temperature = args.temperature
    quant_centers = args.quant_centers
    conv2bohr = args.conv2bohr
    conv_factor = 1.8897259886 if conv2bohr else 1 # conversion factor based on conv2bohr value

    # Convert string quant_centers to 1D numpy array containing the indices
    if not quant_centers:
        quant_centers = quant_centers=np.array([])
    else:
        str_list = quant_centers.split(',')
        int_list = [int(x) for x in str_list]
        quant_centers = np.array(int_list)

    # Store nuclei in symbols list and coordinates in numpy array
    symbols = [] # chemical symbols
    positions = [] # Cartesian coordinates of input geometry
    with open("xfile.txt", "r") as file:
        for line in file:
            words = line.split()
            if len(words) >= 4:
                symbols.append(words[0])
                positions.append([float(words[1]), float(words[2]), float(words[3])])
    positions = conv_factor*np.array(positions)

    # Store energies and gradients in lists
    energies = [0] * nsurf
    gradients = [None] * nsurf
    for i in range(nsurf):
        # Energies
        filename = f'E{i}.txt'
        energy = np.loadtxt(filename, dtype=float)
        energy = energy.item()
        energies[i] = energy
        # Gradients
        filename = f'delE{i}.txt'
        gradient = np.genfromtxt(filename, dtype=float)
        gradients[i] = gradient 
    
    # Store derivative couplings in a list of lists
    dcs = [[[] for _ in range(nsurf)] for _ in range(nsurf)]
    for i in range(nsurf):
        for j in range(i):
            filename = f'd{i}{j}.txt'
            dc = np.genfromtxt(filename, dtype=float)
            dcs[i][j] = dc
            dcs[j][i] = -1*dc # property of derivative couplings
    
    # PRINT RELEVANT READ-IN INFO

    # Print current time
    curr_time = stepnum*dt
    print(f'\n CURRENT TIME: {curr_time:.4f} fs ')

    # Print the nuclear geometry 
    print("\n MOLECULAR GEOMETRY (in bohr): ")
    formatted_positions = [['{:0.10f}'.format(val) for val in row] for row in positions]
    print(np.c_[symbols, formatted_positions])

    print("\n ENERGIES (hartrees), GRADIENTS (hartrees/bohr), & DERIVATIVE COUPLINGS (1/bohr): ")

    # Print energy and gradient info for the user
    for i in range(nsurf):
        print(f'\n E_{i} = {energies[i]:.10f}')
        print(f' Gradient of E_{i}:')
        print(gradients[i])

    # Print derivative coupling info for the user
    for i in range(nsurf):
        for j in range(i):
            print(f'\n Derivative Coupling d{i}{j}:')
            print(dcs[i][j])

    # ******* INITIALIZE & RUN CALCULATION ******* 

    delE0 = gradients[0]
    aimd = AIMD(delE0=delE0, symbols=symbols, positions=positions, nsurf=nsurf, energies=energies, dt=dt, 
                stepnum=stepnum, temperature=temperature, quant_centers=quant_centers, conv2bohr=conv2bohr)
    aimd.run()

if __name__ == "__main__":
    main()