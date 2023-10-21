import numpy as np
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
    
    # USER: DEFINE DYNAMICS PARAMATERS
    nsurf = 2                      # How many surfaces we will consider in the dynamics
    temperature = 100           # Temperature (in Kelvin) used for intializing velocitites 
    neo_centers = np.array([2, 3]) # 1D numpy array of NEO centers
    conv2bohr = True               # Set to true if the Q-Chem input is in Angstrom and we need to convert to bohr

    # Store nuclei in symbols list and coordinates in numpy array
    symbols = [] # chemical symbols
    positions = [] # Cartesian coordinates of input geometry
    with open("geom.txt", "r") as file:
        for line in file:
            words = line.split()
            if len(words) >= 4:
                symbols.append(words[0])
                positions.append([float(words[1]), float(words[2]), float(words[3])])
    factor = 1.8897259886 if conv2bohr else 1 # convert to bohr if necessary
    positions = factor*np.array(positions)

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
    
    # PRINT RELEAVNT READ-IN INFO

    # Print the nuclear geometry 
    print("\n STARTING GEOMETRY (in bohr): ")
    formatted_positions = [['{:0.10f}'.format(val) for val in row] for row in positions]
    print(np.c_[symbols, formatted_positions])

    print("\n ENERGIES, GRADIENTS, & DERIVATIVE COUPLINGS: ")

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

    print("\n ****************** PERFORM NEXT STEP ****************** ")

    delE0 = gradients[0]
    aimd = AIMD(delE0=delE0, symbols=symbols, positions=positions, 
                temperature=temperature, quant_centers=neo_centers)
    aimd.run()

if __name__ == "__main__":
    main()