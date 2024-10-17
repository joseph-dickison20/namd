import os
import numpy as np
from namd.src.tdse import *

# Extract energies
def extract_energy_values(file_path):
    # Initialize empty lists to store E_0 and E_1 values
    e_0_values = []
    e_1_values = []
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the energy values
            if 'E_0' in line:
                # Extract the E_0 value and append to the list
                e_0_values.append(float(line.split('=')[1].strip()))
            elif 'E_1' in line:
                # Extract the E_1 value and append to the list
                e_1_values.append(float(line.split('=')[1].strip()))
    # Convert lists to NumPy arrays
    e_0_array = np.array(e_0_values)
    e_1_array = np.array(e_1_values)
    return e_0_array, e_1_array

# Get energies, NACs, and coeffs
e0, e1 = extract_energy_values('fad_eng.txt')
nacs = np.genfromtxt("fad_nac.txt", dtype=float)
curr_coeffs = np.array([-0.149460 + 0.038859j, -0.943665 + 0.292655j], dtype=complex) # intial

# Store moduli
c0_2 = np.zeros(len(e0))
c1_2 = np.zeros(len(e0))

# Get coeffs
for i in range(len(e0)):
    c0_2[i] = np.abs(curr_coeffs[0]**2)
    c1_2[i] = np.abs(curr_coeffs[1]**2)
    Tmat = np.zeros((2, 2), dtype=float)
    Tmat[0, 1] = nacs[i]
    Tmat[1, 0] = nacs[i]
    energies = [e0[i], e1[i]]
    new_coeffs, _, _ = solve_tdse(curr_coeffs, 0.05, energies, Tmat)
    curr_coeffs = new_coeffs

# Save
np.savetxt('fad_c02.txt', c0_2, fmt='%.6f')
np.savetxt('fad_c12.txt', c1_2, fmt='%.6f')