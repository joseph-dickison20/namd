# Integration of the TSDE, needed in src/get_surfaces.py for both Ehrenfest and FSSH dynamics. 
# We will have to set up a system of ODEs, and we will use the Explicit Runge-Kutta method of order 5(4). 

import numpy as np
from scipy.integrate import RK45 

fs2au = 41.3413733365 # there are this many au of time in one femtosecond, the unit dt is given in

def ode_system(t, y, energies, Tmat, neqs):
    # Refer to Eq. (11) in Yu, Roy, and Hammes-Schiffer
    dci_dts = np.zeros(neqs, dtype=complex)
    for j in range(neqs):
        dci_dt = 0 + 0j  
        for i in range(neqs):
            if j == i:
                dci_dt -= complex(0, energies[i]) * y[i]
            else: 
                dci_dt -= Tmat[j,i] * y[i]
        dci_dts[j] = dci_dt
    return dci_dts

def solve_tdse(coeffs, dt, energies, Tmat):

    """
    Integrates the coupled set of ODEs given in Eq. (11) in Yu, Roy, and Hammes-Schiffer to obtain the new coefficients for each 
    adiabatic state at the end of the classical time step. Derived from substituting the expansion of the wave function as a linear
    combination of adiabatic states into the TDSE and solving for the time derivatives of each coefficient.

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

    # Values needed to initialize the solver
    t0 = 0                         # current time, we always start each classical time step at time zero for the quantum evolution
    t_bound = fs2au*dt             # final time, duration of one classical time step
    y0 = coeffs                    # initial y values
    max_step = 0.01                # a maximum step size of 0.01 au is hardcoded, we do not want any steps larger than this regardless of dt
    neqs = y0.size                 # number of equations in the system of ODEs

    # Initialize the solver
    solver = RK45(fun=lambda t, y: ode_system(t, y, energies, Tmat, neqs), t0=t0, y0=y0, t_bound=t_bound, max_step=max_step)

    # Run the solver and collect time points and solution vectors
    time_points = [t0]  # list of time points
    y_values = [y0]     # list of solutions
    while solver.status == 'running':
        solver.step()
        time_points.append(solver.t)
        solver_y = solver.y / np.linalg.norm(solver.y)
        y_values.append(solver_y)
    
    # The array of new coefficients is the last solution vector of the solver
    new_coeffs = y_values[-1]

    return new_coeffs, time_points, y_values