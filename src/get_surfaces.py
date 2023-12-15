# File containing functions to calculate (Ehrenfest) or choose (FSSH) gradients for classical nuclei to move on.
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
    Tmat = np.zeros((nstates,nstates))
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
        time_points (list): List of points in time (in au) for which the TSDE is integrated
        y_values (list): List of solution vectors at each time point in time_points list
    """

    # Get number of surfaces included
    nsurf = len(energies)

    # Initial print out
    print("\n Coefficients for each adiabatic state at begnning of time step:")
    for i in range(nsurf):
        print(f" C_{i} = {coeffs[i].real:0.6f} + {coeffs[i].imag:0.6f}j, \t |C_{i}|^2 = {(np.abs(coeffs[i])**2):0.6f}")

    # Integrate the TDSE, Eq. (11) in Yu, Roy, and Hammes-Schiffer, to obtain new coefficients
    new_coeffs, time_points, y_values = solve_tdse(coeffs, dt, energies, Tmat)

    # Final print out
    print("\n Coefficients for each adiabatic state at end of time step:")
    for i in range(nsurf):
        print(f" C_{i} = {new_coeffs[i].real:0.6f} + {new_coeffs[i].imag:0.6f}j, \t |C_{i}|^2 = {(np.abs(new_coeffs[i])**2):0.6f}")
    
    return new_coeffs, time_points, y_values

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

def check_hop(time_points, y_values, active_surface, Tmat):

    """
    Calculates whether or not a hop to a new adiabatic state occurs duing a classical time step for FSSH. 

    Args:
        time_points (list): List of points in time (in au) for which the TSDE is integrated
        y_values (list): List of solution vectors at each time point in time_points list
        active_surface (int): Integer indicating which adiabatic state is currently active 
        Tmat (numpy.ndarray): Time-dependent nonadiabatic coupling matrix, T, where T(i,j) = <\psi_i|d/dt(\psi_j)>

    Returns:
        hop_check (int): Integer indicating if hop occured; returns -1 if no hop occured, but if hop occured, hop_check gives the index of the new active state
    """

    # Get the number of states and number of steps
    nstates = Tmat.shape[0]
    nsteps = len(time_points) 

    # Initialize hop_check integer and hop_occured flag
    hop_occurred = False
    hop_check = -1

    # Compute hopping probabilities between the current state and all other states
    for i in range(1,nsteps): # check for all quantum time-steps
        
        # Store the hopping probability between active state and all states
        hop_probs = [] 
        # Calculate hop_prob for all states
        for j in range(nstates): 
            if j != active_surface:
                dtq = time_points[i] - time_points[i-1] # time step
                curr_soln = y_values[i] # the current set of solutions
                mag = np.abs(curr_soln[active_surface])**2 # magnitude of the active coefficient
                # Calculate hop prob between active_state and current state j 
                curr_prob = ((2*dtq)/mag)*np.real(Tmat[active_surface,j]*np.conj(curr_soln[active_surface])*curr_soln[j])
                if curr_prob < 0:
                    hop_probs.append(0)
                else:
                    hop_probs.append(curr_prob)
            else:
                hop_probs.append(0)

        # Sum up the cumulative probabilities 
        cml_probs = [0] + [sum(hop_probs[:k+1]) for k in range(len(hop_probs))]
        rnd = np.random.rand()

        for k in range(len(cml_probs) - 1):
            if cml_probs[k] < rnd < cml_probs[k + 1]:
                hop_occurred = True
                hop_check = k

        # Exit loop if a hop occured
        if hop_occurred:
            break

    return hop_check

def rescale_vels(vels, nacv, masses, ediff):

    """
    Rescales the nuclear velocities if a hop occurs for both a successful and frustrated hop.
    Refer to Jain page 45820, step (viii) for the velocity reversal scheme implemented here.

    Args:
        vels (numpy.ndarray): An array of shape (N, 3) representing nuclear velocities, in bohr/(au of time)
        nacv (numpy.ndarray): An array of shape (N, 3) for the nonadiabtic coupling vector, in 1/bohr
        masses (numpy.ndarray): An array of shape (N,) representing nuclear masses, in m_e
        ediff (float): Energy difference between new active state and current active state, in hartrees

    Returns:
        new_vels (numpy.ndarray): An array of shape (N, 3) representing the new nuclear velocities, in bohr/(au of time)
        frustrated (boolean): Boolean specifying if the hop is frustrated, True if it is, False if it is not
    """

    # Save original error setttings and ignore the divide by zero (for quant_centers)
    original_settings = np.geterr() 
    np.seterr(divide='ignore', invalid='ignore') 

    # Calcualte a, b, and c
    a = 0.5 * np.sum(np.where(masses[:, np.newaxis] != 0, (nacv**2) / masses[:, np.newaxis], 0))
    b = np.sum(vels * nacv)
    c = ediff

    # Calculate gamma
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        print(" HOP FRUSTRATED, REVERSING VELOCITY ")
        frustrated =  True
        gamma = b/a 
    else:
        print(" HOP SUCCESSFUL ")
        frustrated =  False
        gamma = (-1*b + np.sign(b)*np.sqrt(discriminant))/(2*a)

    # Calculate new velocities
    new_vels = vels - gamma*(np.where(masses[:, np.newaxis] != 0, nacv / masses[:, np.newaxis], 0))
    
    # Restore the original error settings
    np.seterr(**original_settings) 

    return new_vels, frustrated