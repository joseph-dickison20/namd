# Integration of the TSDE, needed in src/get_surfaces.py for both Ehrenfest and FSSH dynamics. 
# We will have to set up a system of ODEs, and we will use the Explicit Runge-Kutta method of order 5(4). 

import numpy as np
from scipy.integrate import RK45 

fs2au = 41.3413733365 # there are this many au of time in one femtosecond, the unit dt is given in

def ode_system(t, y, energies, vels, dcs, neqs):
    # Refer to Eq. (11) in Yu, Roy, and Hammes-Schiffer
    dci_dts = np.zeros(neqs, dtype=complex)
    for j in range(neqs):
        dci_dt = 0 + 0j  
        for i in range(neqs):
            if j == i:
                dci_dt -= complex(0, energies[i]) * y[i]
            else: 
                dci_dt -= np.sum(np.multiply(dcs[j][i], vels)) * y[i]
        dci_dts[j] = dci_dt
    return dci_dts

def solve_tdse(coeffs, dt, energies, vels, dcs):

    """
    Integrates the coupled set of ODEs given in Eq. (11) in Yu, Roy, and Hammes-Schiffer to obtain the new coefficients for each 
    adiabatic state at the end of the classical time step. Derived from substituting the expansion of the wave function as a linear
    combination of adiabatic states into the TDSE and solving for the time derivatives of each coefficient.

    Args:
        coeffs (numpy.ndarray): Array of complex numbers giving the coefficients for each adiabat at the beginning of the time step
        dt (float): Time step used to propagate the classical nuclei, in femtoseconds
        energies (list): List of length nsurf containing the energies of the different surfaces, in hartrees 
        vels (numpy.ndarray): An array of shape (N, 3) representing nuclear velocities, in bohr/(au of time)
        dcs (2D list of numpy.ndarray): list indexed by two indices) where dcs[i][j] give the array of shape (N, 3) for the derivative coupling between adiabats i and j, in 1/bohr 

    Returns:
        new_coeffs (numpy.ndarray): Array of complex numbers giving the coefficients for each adiabat at the end of the time step
    """

    # Values needed to initialize the solver
    t0 = 0                         # current time, we always start each classical time step at time zero for the quantum evolution
    t_bound = fs2au*dt             # final time, duration of one classical time step
    y0 = coeffs                    # initial y values
    max_step = 0.02                # a maximum step size of 0.02 au is hardcoded, we do not want any steps larger than this regardless of dt
    neqs = y0.size                 # number of equations in the system of ODEs

    # Initialize the solver
    solver = RK45(fun=lambda t, y: ode_system(t, y, energies, vels, dcs, neqs), t0=t0, y0=y0, t_bound=t_bound, max_step=max_step)

    # Run the solver 
    while solver.status == 'running':
        solver.step()
    
    # The array of new coefficient is the last solution vector of the solver, needs to be normalized
    new_coeffs = solver.y / np.linalg.norm(solver.y)

    return new_coeffs