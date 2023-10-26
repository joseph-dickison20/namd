# File input, output, and printing functions

import os
import numpy as np
from src.sys_info import *

def aimd_print_info(masses, energies, vels, positions, grad, quant_centers):
    
    """
    AIMD printing function.
    """

    # Print quant_centers if applicable
    if np.size(quant_centers) != 0:
        print("\n NOTE the following centers have been quantized and were therefore assigned zero mass and velocity:")
        print(quant_centers)

    # Print positions and velocities
    print("\n VELOCITIES (in bohr/(au of time))")
    print(vels)
    print("\n POSITIONS (in bohr)")
    print(positions)

    # Track COM motion and angular momentum
    com_vel = get_momentum(masses, vels) / np.sum(masses)
    print("\n Velocity of center-of-mass (in bohr/(au of time)):")
    print(com_vel)
    com = get_com(masses, positions)
    adj_positions = positions - com 
    I11, I22, I33, I12, I13, I23 = get_moments_of_inertia(masses, adj_positions)
    Lp = get_angular_momentum(masses, adj_positions, vels, I11, I22, I33)
    print(" Angular momentum of system (in units of hbar):")
    print(Lp)
    
    # Track total energy (in AIMD, potential is just the ground state energy)
    ke = 0.5 * np.sum(masses * np.sum(vels**2, axis=1)) # kinetic energy of system, in hartree
    ke_temp = get_temperature(masses, vels)
    pe = energies[0]
    total_energy = ke + pe
    print(f"\n Kinetic energy of nuclei: {ke:.10f} hartrees, {ke_temp:.2f} Kelvin")
    print(f" Potential energy (energy of surface): {pe:.10f} hartrees")
    print(f" TOTAL ENERGY (KE of nuceli + PE of surface) = {total_energy:.10f} hartrees")


def save_info(symbols, positions, vels, grad, conv2bohr):
    
    """
    Saves relevant info to files in the current working 
    directory that will be read in the next time step.
    """

    # File names and paths
    cwd = os.getcwd()
    xfile = "xfile.txt"
    vfile = "vfile.txt"
    gfile = "gfile.txt"
    file_path_x = os.path.join(cwd, xfile)
    file_path_v = os.path.join(cwd, vfile)
    file_path_g = os.path.join(cwd, gfile)

    # ********* BEGIN SPECIAL FORMAT FOR POSITION FILE *********
    output = []
    new_symbols = [s.replace("Gh", "@H") for s in symbols] # Assume all ghost atoms are H atom centers 
    conv_factor = 1.8897259886 if conv2bohr else 1 # conversion factor based on conv2bohr value
    new_positions = positions/conv_factor # convert from bohr to Angstrom
    for symbol, position in zip(new_symbols, new_positions):
        formatted_position = "\t".join(f"{x:.10f}" for x in position)
        output.append(f"{symbol}\t{formatted_position}")
    final_output = "\n".join(output)
    # ********* END SPECIAL FORMAT FOR POSITION FILE *********

    # Save to file
    with open(file_path_x, "w") as file:
        file.write(final_output)
    np.savetxt(file_path_v, vels, fmt='%.12f', delimiter=' ')
    np.savetxt(file_path_g, grad, fmt='%.12f', delimiter=' ')