# File containing functions to calculate (Ehrenfest) or choose (FSSH) gradeints for classical nuclei to move on.
# Also contains function to obtain new coefficients from the TDSE. The following sources will be cited in the documentation:
#   Qi Yu, Saswata Roy, and Sharon Hammes-Schiffer, Journal of Chemical Theory and Computation 2022 18 (12), 7132-7141, DOI: 10.1021/acs.jctc.2c00938

import numpy as np
from src.tdse import *

def get_new_coeffs(coeffs, dt, energies, vels, dcs):

    """
    Obtains the new coeffcients for each adiabatic state at the end of the classical time step.

    Args:
        coeffs (numpy.ndarray): Array of complex numbers giving the coefficients for each adiabat at the beginning of the time step
        dt (float): Time step used to propagate the classical nuclei, in femtoseconds
        energies (list): List of length nsurf containing the energies of the different surfaces, in hartrees 
        vels (numpy.ndarray): An array of shape (N, 3) representing nuclear velocities, in bohr/(au of time)
        dcs (2D list of numpy.ndarray): list indexed by two indices where dcs[i][j] give the array of shape (N, 3) for the derivative coupling between adiabats i and j, in 1/bohr 

    Returns:
        new_coeffs (numpy.ndarray): Array of complex numbers giving the coefficients for each adiabat at the end of the time step
    """

    # Get number of surfaces included
    nsurf = len(energies)

    # Initial print out
    print("\n Coefficients for each adiabatic state at begnning of time step:")
    for i in range(nsurf):
        print(f" C_{i} = {coeffs[i].real:0.6f} + {coeffs[i].imag:0.6f}j, \t |C_{i}|^2 = {(np.abs(coeffs[i])**2):0.6f}")

    # Integrate the TDSE, Eq. (11) in Yu, Roy, and Hammes-Schiffer, to obtain new coefficients
    new_coeffs = solve_tdse(coeffs, dt, energies, vels, dcs)

    # Final print out
    print("\n Coefficients for each adiabatic state at end of time step:")
    for i in range(nsurf):
        print(f" C_{i} = {new_coeffs[i].real:0.6f} + {new_coeffs[i].imag:0.6f}j, \t |C_{i}|^2 = {(np.abs(new_coeffs[i])**2):0.6f}")

    return new_coeffs

def get_ehrenfest_grad(coeffs, energies, gradients, dcs):

    """
    Calculates the gradient of the average surface for Ehrenfest dynamics.

    Args:
        coeffs (numpy.ndarray): Array of complex numbers giving the time-dependent coefficients for each adiabat
        energies (list): List of length nsurf containing the energies of the different surfaces, in hartrees
        gradients (list of numpy.ndarray): list of arrays of shape (N, 3) giving the gradient of the adiabatic state energies, in hartree/bohr
        dcs (2D list of numpy.ndarray): list indexed by two indices) where dcs[i][j] give the array of shape (N, 3) for the derivative coupling between adiabats i and j, in 1/bohr 

    Returns:
        ehrenfest_grad (numpy.ndarray): An array of shape (N, 3) giving the average surface gradient
    """

    # Preallocate the ehrenfest_grad 
    ehrenfest_grad = np.zeros_like(gradients[0])

    # Use RHS of Eq. (8) in Yu, Roy, and Hammes-Schiffer to construct the gradient
    # Note the negative sign is ignored since it will accounted for in the Verlet
    nsurf = len(energies)
    for i in range(nsurf):
        for j in range(nsurf):
            weight = np.abs(coeffs[i]*coeffs[j])
            if i == j:
                grad_term = gradients[i]
            else:
                ediff = energies[j] - energies[i]
                grad_term = ediff*dcs[i][j]
            ehrenfest_grad += weight*grad_term

    return ehrenfest_grad