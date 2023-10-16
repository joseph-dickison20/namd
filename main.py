import numpy as np
from src.calc_classes import *

def main():
    
    # ******************************
    #       MAIN RUN FUNCTION
    # ******************************

    # This main.py script is the interface bewteen your ES code
    # and the dynamics code in the src/ directory updating the 
    # position of classical nuclei via AIMD, Ehrenfest, or FSSH.

    # See the README.txt for details on the workflow. 

    # ******* READ IN & DEFINE NECESSARY PARAMETERS ******* 
    
    # USER: DEFINE DYNAMICS PARAMATERS
    nsurf = 2 # How many surfaces we will consider in the dynamics

    # Store nuclei in symbols list and coordinates in NumPy array
    symbols = [] # chemical symbols
    coords = [] # Cartesian coordinates of input geometry
    with open("geom.txt", "r") as file:
        for line in file:
            words = line.split()
            if len(words) >= 4:
                symbols.append(words[0])
                coords.append([float(words[1]), float(words[2]), float(words[3])])
    coords = np.array(coords)
    ncen = coords.shape[0] # number of basis function centers in molecule

    # Print the nuclear geometry
    print("\n STARTING GEOMETRY: \n ")
    print(np.c_[symbols, coords])

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
    
    # Print extracted info
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
    print("\n We would now set up and initialize the calculation here.\n")

if __name__ == "__main__":
    main()