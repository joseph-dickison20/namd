import os
import numpy as np
import argparse
from namd.src.calc_classes import *

def main():
    
    # **********************************************************
    # ****************   MAIN NAMD INTERFACE   *****************
    # **********************************************************

    # This main.py script is the interface bewteen your ES code
    # and the dynamics code in the src/ directory, updating the 
    # position of classical nuclei via AIMD, Ehrenfest, or FSSH.

    # See the README.txt for details on the workflow. 

    
    # **************** READ IN & DEFINE NECESSARY PARAMETERS **************** 
    
    # Parse arguments, set defaults
    parser = argparse.ArgumentParser(description="Argument parser for NAMD module")
    parser.add_argument("--nsurf", type=int, default=1, help="How many surfaces to consider in the dynamics")
    parser.add_argument("--dt", type=float, default=0.06, help="Time step (in femtoseconds)")
    parser.add_argument("--stepnum", type=int, default=0, help="The current step number of the trajectory")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature (in Kelvin) used for intializing velocitites")
    parser.add_argument("--td_coeffs", type=str, default="", help="String with an nsurf number of inital coefficients for either Ehrenfest of FSSH")
    parser.add_argument("--quant_centers", type=str, default="", help="String of nuclear indices that are quantized, place commas in between (start count at 0)")
    parser.add_argument("--fixed_centers", type=str, default="", help="String of nuclear indices that are fixed, place commas in between (start count at 0)")
    parser.add_argument("--conv2bohr", action='store_true', help="Controls whether or not to convert Cartesian coodfinates from Angstrom to bohr (include for true, exclude for false)")
    parser.add_argument("--num_TDNAC", action='store_true', help="Controls whether or not the TD-NAC will be calculated numerically (include flag if true, exclude if false)")
    parser.add_argument("--vel_init", action='store_true', help="Controls whether or not the user will provide inital velocities (include flag if true, exclude if false)")
    args = parser.parse_args()

    # Obtain values from arguments
    dt = args.dt
    nsurf = args.nsurf
    stepnum = args.stepnum
    temperature = args.temperature
    td_coeffs = args.td_coeffs
    quant_centers = args.quant_centers
    fixed_centers = args.fixed_centers
    conv2bohr = args.conv2bohr
    num_TDNAC = args.num_TDNAC
    vel_init = args.vel_init

    
    # **************** CONVERT READ-IN PARAMETERS TO USABLE FORMS ****************

    # Account for conversion factor to bohr
    conv_factor = 1.8897259886 if conv2bohr else 1 

    # Convert string quant_centers to 1D numpy array containing the indices
    if not quant_centers:
        quant_centers = quant_centers=np.array([])
    else:
        str_list = quant_centers.split(',')
        int_list = [int(x) for x in str_list]
        quant_centers = np.array(int_list)

    # Convert string fixed_centers to 1D numpy array containing the indices
    if not fixed_centers:
        fixed_centers = fixed_centers=np.array([])
    else:
        str_list = fixed_centers.split(',')
        int_list = [int(x) for x in str_list]
        fixed_centers = np.array(int_list)
    
    # Convert string of td_coeffs to 1D numpy arrary containing the initial coefficients or load them in
    if stepnum == 0:
        if not td_coeffs:
            td_coeffs = np.array([])
        else:
            str_list = td_coeffs.split(',')
            complex_list = [complex(x) for x in str_list]
            td_coeffs = np.array(complex_list)
            td_coeffs = td_coeffs / np.sqrt(np.sum(td_coeffs**2))
    else:
        td_coeffs = np.genfromtxt('tdfile.txt', dtype=complex)

    # Find active surface for FSSH
    if stepnum == 0:
        # Assume we begin in one adiabatic state, not a coherent mixture
        active_surface = int(np.where(np.real(td_coeffs) == 1)[0][0]) if np.any(np.real(td_coeffs) == 1) else None
    else:
        if os.path.exists("active_surface.txt"):
            with open("active_surface.txt", "r") as file:
                active_surface = int(file.read())

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
    
    
    # **************** PRINT RELEVANT INFO ****************

    # Print current time
    curr_time = stepnum*dt
    print(f'\n CURRENT TIME: {curr_time:.4f} fs ')
    if stepnum == 0:
        print("\n NOTE: For all methods, the inital time step will not integrate the TDSE.")
        print(" The nuclei will move on the input surface for one step.")
    
    # Print the nuclear geometry 
    print("\n MOLECULAR GEOMETRY (in bohr): ")
    formatted_positions = [['{:0.10f}'.format(val) for val in row] for row in positions]
    print(np.c_[symbols, formatted_positions])

    print("\n READ-IN ENERGIES (hartrees), GRADIENTS (hartrees/bohr), & DERIVATIVE COUPLINGS (1/bohr): ")

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

    
    # **************** INITIALIZE & RUN CALCULATION **************** 
    
    """
    # AIMD 
    aimd = AIMD(delE0=gradients[0], symbols=symbols, positions=positions, nsurf=nsurf, 
                energies=energies, dt=dt, stepnum=stepnum, temperature=temperature, 
                quant_centers=quant_centers, fixed_centers=fixed_centers, 
                conv2bohr=conv2bohr, vel_init=vel_init)
    aimd.run()
    """

    """
    # Ehrenfest
    ehrenfest = Ehrenfest(gradients=gradients, dcs=dcs, td_coeffs=td_coeffs, num_TDNAC=num_TDNAC, 
                          symbols=symbols, positions=positions, nsurf=nsurf, energies=energies, dt=dt, 
                          stepnum=stepnum, temperature=temperature, quant_centers=quant_centers, 
                          fixed_centers=fixed_centers, conv2bohr=conv2bohr, vel_init=vel_init)
    ehrenfest.run()
    """
    
    # FSSH
    fssh = FSSH(gradients=gradients, active_surface=active_surface, dcs=dcs, td_coeffs=td_coeffs, 
                num_TDNAC=num_TDNAC, symbols=symbols, positions=positions, nsurf=nsurf, energies=energies, 
                dt=dt, stepnum=stepnum, temperature=temperature, quant_centers=quant_centers, 
                fixed_centers=fixed_centers, conv2bohr=conv2bohr, vel_init=vel_init)
    fssh.run()

if __name__ == "__main__":
    main()