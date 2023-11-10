# File containing functions to calculate (Ehrenfest) or choose (FSSH) gradeints for classical nuclei to move on.
# Also contains function to obtain new coefficients from the TDSE. The following sources will be cited in the documentation:
#   Qi Yu, Saswata Roy, and Sharon Hammes-Schiffer, Journal of Chemical Theory and Computation 2022 18 (12), 7132-7141, DOI: 10.1021/acs.jctc.2c00938
#   Amber Jain, Ethan Alguire, and Joseph E. Subotnik, Journal of Chemical Theory and Computation 2016 12 (11), 5256-5268, DOI: 10.1021/acs.jctc.6b00673

import numpy as np
from scipy.linalg import logm, sqrtm
from src.tdse import *

fs2au = 41.3413733365 # there are this many au of time in one femtosecond, the unit dt is given in

def get_T_matrix(vels, dcs, dt, num_TDNAC):

    """
    Calculates the time-dependent nonadiabatic coupling matrix, T, where T(i,j) = <\psi_i|d/dt(\psi_j)>.
    Algorithm for both numerical and analytically calcualting this matrix is included.

    Args:
        vels (numpy.ndarray): An array of shape (N, 3) representing nuclear velocities, in bohr/(au of time)
        dcs (2D list of numpy.ndarray): List indexed by two indices where dcs[i][j] give the array of shape (N, 3) for the derivative coupling between adiabats i and j, in 1/bohr 
        dt (float): Time step used to propagate the classical nuclei, in femtoseconds
        num_TDNAC (boolean): True if user is calculating the TD-NAC matrix numerically

    Returns:
        Tmat (numpy.ndarray): Time-dependent nonadiabatic coupling matrix, T, where T(i,j) = <\psi_i|d/dt(\psi_j)>
    """

    # Preallocate and prepare and Tmat computation
    nstates = len(dcs) # the number of states included in the expansion
    Tmat = np.ones((nstates,nstates))
    dt *= fs2au # convert time step to au of time

    # Calculate Tmat numerically or analytically based on user input
    if num_TDNAC:  
        # Numercial calculation
        # Refer to Eq. (29) and (34) of Jain et al. 
        Umat = np.genfromtxt('TD_NAC.txt', dtype=float) # User-provided nstates-by-nstates matrix where U(i,j) = <\psi_i(t0)|\psi_j(t-+dt)>
        Umat = Umat@sqrtm(Umat.T@Umat) # Lowdin orthogonolization
        Tmat = (1/dt)*logm(Umat) # generalized Meek and Levine expression
    else:           
        # Analytical calculation
        # Refer to Eq. (12) of Yu et al.
        for j in range(nstates):
            for i in range(nstates):
                if i != j:
                    Tmat[j,i] = np.sum(np.multiply(dcs[j][i], vels)) # Tji = dot(v,dji)

    return Tmat

def get_new_coeffs(coeffs, dt, energies, Tmat):

    """
    Obtains the new coeffcients for each adiabatic state at the end of the classical time step.

    Args:
        coeffs (numpy.ndarray): Array of complex numbers giving the coefficients for each adiabat at the beginning of the time step
        dt (float): Time step used to propagate the classical nuclei, in femtoseconds
        energies (list): List of length nsurf containing the energies of the different surfaces, in hartrees 
        Tmat (numpy.ndarray): Time-dependent nonadiabatic coupling matrix, T, where T(i,j) = <\psi_i|d/dt(\psi_j)>

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
    new_coeffs = solve_tdse(coeffs, dt, energies, Tmat)

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
            weight = np.abs(np.conj(coeffs[i])*coeffs[j])
            if i == j:
                grad_term = gradients[i]
            else:
                ediff = energies[j] - energies[i]
                grad_term = ediff*dcs[i][j]
            ehrenfest_grad += weight*grad_term

    return ehrenfest_grad