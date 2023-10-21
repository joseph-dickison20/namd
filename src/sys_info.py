# A set of functions to obtain releavnt quantities about the system, such as mass, temperature, etc.
# Refer to src/calc_classes.py and src/velocity_funcs.py for argument types and formats.

import numpy as np

kb = 3.166811563e-6 # Boltzmann constant in hartree/Kelvin

def get_masses(symbols, quant_centers):
    # Dictionary to map chemical symbols to amu
    symbol_to_mass = {
        "H": 1.00797,
        "C": 12.011,
        "O": 15.9994,
        "F": 18.998403,
        "Gh": 0 # ghost atom
    }
    # Convert symbols to masses using the dictionary and get masses
    mass_list = [symbol_to_mass.get(item, 0) for item in symbols]
    masses = 1822.88848*np.array(mass_list) # there are 1822.88848 m_e in one amu
    # Account for quantized centers
    for i in quant_centers:
        masses[i] = 0
    return masses

def get_temperature(masses, vels):
    kin = 0.5 * np.sum(masses * np.sum(vels**2, axis=1)) # kinetic energy of system, in hartree
    temperature = (2 * kin) / (3 * kb * np.count_nonzero(masses)) # equipartition theorem 
    return temperature

def get_com(masses, positions): 
    return (np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses))

def get_momentum(masses, vels):
    return np.sum(masses[:, np.newaxis] * vels, axis=0)

def get_com_vel(masses,vels):
    return get_momentum(masses, vels) / np.sum(masses)

def get_moments_of_inertia(masses, positions):
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(positions)):
        x, y, z = positions[i]
        m = masses[i]
        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z
    return I11, I22, I33, I12, I13, I23

def get_angular_momentum(masses, positions, vels, I11, I22, I33):
    angular_momenta = np.cross(positions, vels * masses[:, np.newaxis])
    Lx = np.sum(angular_momenta[:, 0])
    Ly = np.sum(angular_momenta[:, 1])
    Lz = np.sum(angular_momenta[:, 2])
    return np.array([Lx, Ly, Lz])

def get_rotation_vector(I11, I22, I33, Lp):
    Ixx, Iyy, Izz = I11, I22, I33
    omega_x = Lp[0] / Ixx if Ixx > 0 else 0
    omega_y = Lp[1] / Iyy if Iyy > 0 else 0
    omega_z = Lp[2] / Izz if Izz > 0 else 0
    return np.array([omega_x, omega_y, omega_z])

def rescale_vel_to_temp(masses, vels, desired_temp):
    curr_temp = get_temperature(masses,vels)
    rescale_vels = np.sqrt(desired_temp/curr_temp)*vels
    return rescale_vels
