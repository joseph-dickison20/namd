# Set of functions for initializing velcoities of the system. 
# An inital set of velocities is selected based on a Maxwell-Boltzmann distribution. 
# Linear momentum of center of mass is forced to zero and rigid rotational motion
# must be accounted at the inital time step. Velocities are rescaled to match 3/2NkT at the initial step,
# but velocities should not be rescaled after the initlialization (NVE).

import numpy as np
from namd.sys_info import *

kb = 3.166811563e-6 # Boltzmann constant in hartree/Kelvin

def stationary_com(masses, vels):

    """
    Sets the total linear momentum to zero to counteract translational motion.

    Args:
        masses (numpy.ndarray): An array of shape (N,) representing nuclear masses, in m_e.
        vels (numpy.ndarray): An array of shape (N, 3) representing nuclear velocities, in bohr/(au of time)

    Returns:
        new_vels (numpy.ndarray): Adjusted velocities after translational motion.
    """

    # Calculate total momentum and center-of-mass velocity
    com_vel = get_momentum(masses, vels) / np.sum(masses)
    
    # All centers with no velocity are centers with no mass
    # Therefore, we only subtract com_vel from centers with nonzero velocity
    all_zero_rows = np.all(vels == 0, axis=1)
    new_vels = np.where(all_zero_rows[:, np.newaxis], vels, vels - com_vel)

    return new_vels

def no_rotation(masses, positions, vels):
    
    """
    Sets the total angular momentum to zero to counteract rigid rotations.

    Args:
        masses (numpy.ndarray): An array of shape (N,) representing nuclear masses, in m_e.
        positions (numpy.ndarray): An array of shape (N, 3) representing nuclear positions, in bohr
        vels (numpy.ndarray): An array of shape (N, 3) representing nuclear velocities, in bohr/(au of time)

    Returns:
        new_vels (numpy.ndarray): Adjusted velocities after counteracting rigid rotations.
    """

    # Calculate the center of mass
    com = get_com(masses, positions)
    
    # Translate positions to move the center of mass to the origin
    positions -= com
    
    # Calculate the moments of inertia
    I11, I22, I33, I12, I13, I23 = get_moments_of_inertia(masses, positions)
    
    # Calculate the total angular momentum and transform it to the principal basis
    Lp = get_angular_momentum(masses, positions, vels, I11, I22, I33)

    # Calculate the rotation velocity vector in the principal basis
    omega = get_rotation_vector(I11, I22, I33, Lp)
    
    # Adjust velocities to counteract rigid rotations
    all_zero_rows = np.all(vels == 0, axis=1)
    new_vels = np.where(all_zero_rows[:, np.newaxis], vels, vels - np.cross(omega, positions))

    return new_vels

def get_initial_vels(masses, positions, temperature):
    
    """
    Calculates an initial set of velocities for the trajectory according to the 
    Maxwell-Boltzmann distribution. Translational and rotational motion is accounted
    for, but only at the first time step. Velocities are scaled to match temperature
    given by the equipartition theorem.

    Args:
        masses (numpy.ndarray): An array of shape (N,) representing nuclear masses, in m_e.
        positions (numpy.ndarray): An array of shape (N, 3) representing nuclear positions, in bohr
        temperature (float): temperature to set velocities, in Kelvin

    Returns:
        vel_inits (numpy.ndarray): Initial set of velocities in bohr/(au of time)
    """

    # The M-B distribution is Gaussian for each Cartesian direction
    # Pick random velocities in each direction, account for if a center has no mass by giving it no velocity
    ncen = len(masses) # how many centers we have
    vel_inits = np.empty((ncen, 3))
    for i in range(ncen):
        for j in range(3):
            mass = masses[i]
            if mass == 0:
                vel_inits[i, j] = 0
            else:
                sigma = np.sqrt(kb*temperature/mass) # sigma for M-B distribution
                vel_inits[i, j] = np.random.normal(scale=sigma)
    
    # Rescale velocities to match the desired temperature
    vel_inits = rescale_vel_to_temp(masses, vel_inits, temperature)

    # Set the center-of-mass (COM) velocity to zero 
    vel_inits = stationary_com(masses, vel_inits)
    vel_inits = rescale_vel_to_temp(masses, vel_inits, temperature)

    # Counteract rigid rotations and rescale
    vel_inits = no_rotation(masses, positions, vel_inits)
    vel_inits = rescale_vel_to_temp(masses, vel_inits, temperature)

    return vel_inits
