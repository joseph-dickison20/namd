# Verification of NAMD FSSH code with Tully Models  

# The objecive of this script is to verify the NAMD FSSH code using Tully's model systems he introduces in
# Tully, J. C. Molecular Dynamics with Electronic Transitions. The Journal of Chemical Physics 1990, 93 (2), 1061–1071. https://doi.org/10.1063/1.459170.

import numpy as np
import matplotlib.pyplot as plt

#############################################################################################
# COMPUTE ENERGIES AND COUPLINGS
# EXAMPLE USAGE OF THE BELOW COMPUTE AND MAKE_PLOT FUNCTIONS:
# To compute for a specific x:
#   V1, V2, d12 = compute(x=5.0, A=0.01, B=1.6, C=0.005, D=1.0, E0=0.05, model_num=1)
#   print(f"At x=5.0: V1={V1}, V2={V2}, d12={d12}")
# To compute for a specific x with default parameters:
#   V1, V2, d12 = compute(5.0)
#   print(f"At x=5.0: V1={V1}, V2={V2}, d12={d12}")
# To generate a plot over a range of x values:
#   make_plot(np.linspace(-10, 10, 500), A=0.01, B=1.6, C=0.005, D=1.0, E0=0.05, model_num=1)
# To generate a plot for all three models with their default parameters:
#   make_plot()
#   make_plot(A=0.10, B=0.28, C=0.015, D=0.06, E0=0.05, model_num=2)
#   make_plot(A=0.0006, B=0.10, C=0.90, model_num=3)
# Note the default values all correspond to using Tully's simple avoided crossing model
#############################################################################################

def compute(x, A=0.01, B=1.6, C=0.005, D=1.0, E0=0.05, model_num=1, delta=1e-5):
    """
    Compute the adiabatic potentials (V1, V2), the nonadiabatic coupling (d12),
    and the gradient of V1 and V2 with respect to x using finite difference.
    
    Parameters:
    - x: Input x value
    - A, B, C, D, E0: Model parameters
    - model_num: 1 = Simple avoided crossing
                 2 = Dual avoided crossing (includes E0)
                 3 = Extended coupling with reflection
    - delta: Small value used for finite difference approximation of gradients
    """
    
    # First, compute V1, V2, and d12 at the input value of x
    V1, V2, d_12 = adiabatic_potentials_and_coupling(x, *get_potential_functions(A, B, C, D, E0, model_num))

    # Now compute V1 and V2 at x + delta and x - delta to compute their gradients
    V1_plus, V2_plus, _ = adiabatic_potentials_and_coupling(x + delta, *get_potential_functions(A, B, C, D, E0, model_num))
    V1_minus, V2_minus, _ = adiabatic_potentials_and_coupling(x - delta, *get_potential_functions(A, B, C, D, E0, model_num))

    # Compute the gradients using finite difference
    grad_V1 = (V1_plus - V1_minus) / (2 * delta)
    grad_V2 = (V2_plus - V2_minus) / (2 * delta)
    
    # Return computed values
    return V1, V2, d_12, grad_V1, grad_V2

def get_potential_functions(A, B, C, D, E0, model_num):
    """
    Helper function to return the potential and derivative functions for the chosen model.
    """
    if model_num == 1:
        V11_func = lambda x: A * (1 - np.exp(-B * x)) if x >= 0 else -A * (1 - np.exp(B * x))
        V12_func = lambda x: C * np.exp(-D * x**2)
        V22_func = lambda x: -V11_func(x)
        dV11_dx_func = lambda x: A * B * np.exp(-B * x) if x >= 0 else A * B * np.exp(B * x)
        dV12_dx_func = lambda x: -2 * D * x * C * np.exp(-D * x**2)
        dV22_dx_func = lambda x: -dV11_dx_func(x)

    elif model_num == 2:
        V11_func = lambda x: 0
        V12_func = lambda x: C * np.exp(-D * x**2)
        V22_func = lambda x: -A * np.exp(-B * x**2) + E0
        dV11_dx_func = lambda x: 0
        dV12_dx_func = lambda x: -2 * D * x * C * np.exp(-D * x**2)
        dV22_dx_func = lambda x: 2 * A * B * x * np.exp(-B * x**2)
        
    elif model_num == 3:
        V11_func = lambda x: A
        V12_func = lambda x: B * (2 - np.exp(-C * x)) if x >= 0 else B * np.exp(C * x)
        V22_func = lambda x: -A
        dV11_dx_func = lambda x: 0
        dV12_dx_func = lambda x: B * C * np.exp(-C * x) if x >= 0 else B * C * np.exp(C * x)
        dV22_dx_func = lambda x: 0

    return V11_func, V12_func, V22_func, dV11_dx_func, dV12_dx_func, dV22_dx_func

def make_plot(x_range=np.linspace(-10, 10, 500), A=0.01, B=1.6, C=0.005, D=1.0, E0=0.05, model_num=1):
    """
    Generate a plot of the adiabatic potentials (V1, V2) and nonadiabatic coupling (d12)
    over a specified range of x values, using parameters A, B, C, D, E0 (used in model 2), and model number.
    """
    # Arrays to store results
    V1_values = []
    V2_values = []
    d12_values = []

    # Loop over x values and compute potentials and nonadiabatic coupling
    for x in x_range:
        # Capture only V1, V2, and d12 (ignore gradients in this case)
        V1, V2, d12, _, _ = compute(x, A, B, C, D, E0, model_num)
        V1_values.append(V1)
        V2_values.append(V2)
        d12_values.append(d12)

    # Convert lists to numpy arrays
    V1_values = np.array(V1_values)
    V2_values = np.array(V2_values)
    d12_values = np.array(d12_values)

    # Plot the adiabatic potentials and nonadiabatic coupling on the same plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # First y-axis (left) for adiabatic potentials
    ax1.set_xlabel(r'$x$ (au)')
    ax1.set_ylabel('Adiabatic Potentials (au)', color='tab:blue')
    ax1.plot(x_range, V1_values, label='V1 (Lower Adiabatic Surface)', color='blue')
    ax1.plot(x_range, V2_values, label='V2 (Upper Adiabatic Surface)', color='red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Second y-axis (right) for nonadiabatic coupling
    ax2 = ax1.twinx()
    ax2.set_ylabel('Nonadiabatic Coupling Strength (au$^{-1}$)', color='tab:green')
    ax2.plot(x_range, d12_values, label='Nonadiabatic Coupling (d12)', color='green', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Create a combined legend at the bottom of the plot
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=True, ncol=3)

    # Title
    plt.title('Adiabatic Potentials and Nonadiabatic Coupling Strength')

    # Show plot
    plt.tight_layout()
    plt.show()

def adiabatic_potentials_and_coupling(x, V11_func, V12_func, V22_func, dV11_dx_func, dV12_dx_func, dV22_dx_func):
    """Diagonalizes the diabatic potential matrix to get adiabatic potentials and nonadiabatic coupling."""
    # Diabatic potential matrix
    V = diabatic_potentials(x, V11_func, V12_func, V22_func)
    
    # Diagonalize the matrix to get the adiabatic potentials (eigenvalues) and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(V)
    
    # Ensure eigenvalues are sorted in ascending order and rearrange eigenvectors accordingly
    sorted_indices = np.argsort(eigvals)  # Indices that will sort the eigenvalues
    eigvals = eigvals[sorted_indices]     # Sort the eigenvalues
    eigvecs = eigvecs[:, sorted_indices]  # Reorder the eigenvectors to match sorted eigenvalues
    
    # Ensure consistent phase for eigenvectors by checking the first component of each
    if eigvecs[0, 0] < 0:  # If the first component of the first eigenvector is negative
        eigvecs[:, 0] *= -1  # Flip the sign of the first eigenvector
    if eigvecs[0, 1] < 0:  # If the first component of the second eigenvector is negative
        eigvecs[:, 1] *= -1  # Flip the sign of the second eigenvector
    
    # The sorted eigenvalues are the adiabatic potentials
    V1, V2 = eigvals
    
    # Compute the derivative of the diabatic potential matrix elements
    dV_dx = diabatic_potential_derivative(x, dV11_dx_func, dV12_dx_func, dV22_dx_func)
    
    # Nonadiabatic coupling element between the two adiabatic states
    v1 = eigvecs[:, 0]  # First eigenvector (corresponding to V1)
    v2 = eigvecs[:, 1]  # Second eigenvector (corresponding to V2)
    
    if V2 != V1:
        d_12 = (v1.T @ dV_dx @ v2) / (V2 - V1)
    else:
        d_12 = 0
    
    # Return the potentials and nonadiabatic coupling
    return V1, V2, d_12

def diabatic_potentials(x, V11_func, V12_func, V22_func):
    """Computes the diabatic potential matrix elements using provided functions."""
    V11 = V11_func(x)
    V12 = V12_func(x)
    V22 = V22_func(x)
    return np.array([[V11, V12], [V12, V22]])

def diabatic_potential_derivative(x, dV11_dx_func, dV12_dx_func, dV22_dx_func):
    """Computes the derivative of the diabatic potential matrix elements with respect to x."""
    dV11_dx = dV11_dx_func(x)
    dV12_dx = dV12_dx_func(x)
    dV22_dx = dV22_dx_func(x)
    return np.array([[dV11_dx, dV12_dx], [dV12_dx, dV22_dx]])

#############################################################################################################################
# CHECK TRANSMISSION PROBABILITIES
# In the above block of code, we have functions that compute all the necessary quantities needed to perform the dynamics.
# Below, we use the NAMD FSSH code in order to verify that it gives anticipated results. Below, we run a ntraj number of 
# independent trajectories for the default single avoided crossing model. At the initial momentum value p_init below (20 au),
# according to Tully's Figure 4 in the previously cited paper, we should see ~50% of trajectories end in the ground and 
# excited state, each. One can test over a range of p_init values (the same ones that Tully does in Figures 4, 5, and 6)
# to repoduce the results of each of these figures. However, this will only return the proportion of trajectories 
# tramsitted on the ground and excited states for one given value of p_init. Note that ntraj is kept small, since 
# execution of this code may take a long time. If you want to run more extensive calculations and repoduce all of Tully's 
# figures, you should proabbly submit this calcualtion to a cluster and test over the desired range of p_init values.
##############################################################################################################################
"""
from namd.calc_classes import FSSH
from namd.sys_info import get_temperature

# Define parameters for Tully Model
A = 0.01
B = 1.6
C = 0.005
D = 1.0
E0 = 0.05 
model_num = 1
ntraj = 6

# Run a ntraj number of trajectories
transmit_ground = 0
transmit_excited = 0
for i in range(ntraj):
    
    # Initial conditions
    x_init = -4.0  # Initial position (a.u.)
    p_init = 20.0 # Initial momentum (a.u.)
    mass = 1822.88848*1.00797 # Mass of particle (a.u.), assuming hydrogen

    # Compute the initial velocity
    v_init = p_init / mass  # a.u. for velocity

    # Initial positions (3D system, but keeping everything on x-axis)
    positions = np.array([[x_init, 0.0, 0.0]])  # y, z components are 0

    # Initial velocities
    velocities = np.array([[v_init, 0.0, 0.0]])
    np.savetxt('vfile.txt', velocities, fmt='%.10f') # save to vfile.txt
    temp_tmp = get_temperature(np.array([mass]), velocities) # have to save current temperature since step zero does a mandatory rescaling

    # Prepare FSSH-specific parameters (computed below)
    gradients = None  # Placeholder until computed
    active_surface = 0  # Start on the lower adiabatic state
    td_coeffs = np.array([1.0 + 0j, 0j])  # Start with full population in state 0
    num_TDNAC = False  # Non-adiabatic coupling is computed, not numerical

    # Time step size
    dt = 0.005  # Time step in fs
    total_steps = 4000  # Total number of time steps

    # Set up arrays to store positions and velocities and coefficients squared
    positions_over_time = [x_init]
    velocities_over_time = [v_init]
    c02 = [1]
    c12 = [0]

    # Helper function to compute new gradients, energies, and nonadiabatic couplings
    def compute_quantities(x, A, B, C, D, E0, model_num):
        V1, V2, d12, grad_V1, grad_V2 = compute(x, A, B, C, D, E0, model_num)
        energies = [V1, V2]  # Energies of lower and upper states
        gradients = [np.array([grad_V1, 0, 0]), np.array([grad_V2, 0, 0])] # Gradients of the potential energies
        dcs = [[0, np.array([d12, 0, 0])], [-1 * np.array([d12, 0, 0]), 0]] # Nonadiabatic couplings matrix (antisymmetric)
        return energies, gradients, dcs

    # Initial computation of energies, gradients, and nonadiabatic couplings
    energies, gradients, dcs = compute_quantities(x_init, A, B, C, D, E0, model_num)
    np.savetxt('gfile.txt', gradients[0], fmt='%.10f') # save to gfile.txt

    # Loop to run the simulation for the specified number of steps
    for step in range(total_steps):
        
        # Define a new FSSH object for each time step with the updated values
        fssh_simulation = FSSH(
            gradients=gradients, 
            active_surface=active_surface, 
            dcs=dcs, 
            td_coeffs=td_coeffs, 
            num_TDNAC=num_TDNAC, 
            symbols=["H"],              # Assuming Hydrogen (make sure mass above matches)
            positions=positions, 
            nsurf=2, 
            energies=energies, 
            dt=dt, 
            stepnum=step,      
            temperature=temp_tmp,       # Using the initial temperarture so the rescaling does not alter the velocity
            quant_centers=np.array([]), 
            fixed_centers=np.array([]), 
            conv2bohr=False, 
            vel_init=True
        )
        
        # Run one step of the FSSH dynamics
        fssh_simulation.run()

        # Read updated position from xfile.txt
        with open('xfile.txt', 'r') as xfile:
            x_data = xfile.readline().strip().split()
            new_x = float(x_data[1])  # Assuming new_x is in the second position
        
        # Read updated velocity from vfile.txt
        with open('vfile.txt', 'r') as vfile:
            v_data = vfile.readline().strip().split()
            new_velocity = float(v_data[0])  # Assuming new velocity is the first value

        # Read updated active surface from active_surface.txt
        with open('active_surface.txt', 'r') as active_surface_file:
            active_surface = int(active_surface_file.readline().strip())

        # Read updated coefficients
        td_coeffs = np.genfromtxt('tdfile.txt', dtype=complex)

        # Update positions and velocities
        positions = np.array([[new_x, 0.0, 0.0]])  # Assuming only x-component is updated, with y and z as zero
        velocities = np.array([[new_velocity, 0.0, 0.0]])  # Assuming only x-component is updated, with y and z as zero

        # Compute new energies, gradients, and nonadiabatic couplings based on the new position
        energies, gradients, dcs = compute_quantities(new_x, A, B, C, D, E0, model_num)

        # Append new positions and velocities and coefficients to the lists
        positions_over_time.append(new_x)  # Append the first (and only) position
        velocities_over_time.append(new_velocity)  # Append the first (and only) velocity
        c02.append(np.abs(td_coeffs[0])**2)
        c12.append(np.abs(td_coeffs[1])**2)

    # CHANGE THESE CONDITIONS IF USING SOME OTHER MODEL BESIDES DEFAULT
    if (new_x > 4.0 and active_surface == 0): 
        transmit_ground = transmit_ground + 1
    elif (new_x > 4.0 and active_surface == 1):
        transmit_excited = transmit_excited + 1

    # Plot the position and square coefficients over time (along the x-axis)
    #time_steps = np.arange(total_steps+1) * dt  # Time in fs

    #plt.figure(figsize=(8, 6))
    #plt.plot(time_steps, positions_over_time, label='Position along x-axis')
    #plt.xlabel('Time (fs)')
    #plt.ylabel('Position (a.u.)')
    #plt.title('FSSH Trajectory: Position vs Time')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    #plt.figure(figsize=(8, 6))
    #plt.plot(time_steps, c02, label=r'|C_0|^2', color = 'r')
    #plt.plot(time_steps, c12, label=r'|C_1|^2', color = 'b')
    #plt.xlabel('Time (fs)')
    #plt.ylabel(r'|C_i|^2 (a.u.)')
    #plt.title('FSSH Trajectory: Amplitudes vs Time')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

print("\n")
print(f" Proportion of trajectories tramsitted on ground state = {(transmit_ground/ntraj):0.10f}")
print(f"Proportion of trajectories tramsitted on excited state = {(transmit_excited/ntraj):0.10f}")
"""
#########################################################################################################################
# CHECK FLUX CONDITION
# Another condition that will be useful to check is if the flux condition is satified. The flux condition is as follows:
#   Suppose, for a two state problem, that the current ampltiude of the active state is X. If over the course of a time 
#   step, the amplitude of that active state changes to Y, then the FFSH algorithm should predict that a NSWITCH number
#   of trajectories out of a NTRAJ number of total trajecotries should switch to state Y, with NSWTICH given by
#       NSWITCH = (X-Y)/X*NTRAJ
# Below, we use the start the particle in a region of strong coupling with a high momentum, and begin it in the ground
# state. This will cause a good amount of probability to transfer to the excited state. Note that beginning in the ground
# state is hardcoded, but one can easily modify the below code to have different initial conditions.
#########################################################################################################################

from namd.calc_classes import FSSH
from namd.sys_info import get_temperature

# Define parameters for Tully Model
A = 0.01
B = 1.6
C = 0.005
D = 1.0
E0 = 0.05 
model_num = 1

# Loop to run the simulation for the specified number of steps
ntraj = 1000
num_switch = 0

for i in range(ntraj):

    # Initial conditions
    x_init = -0.0005  # Initial position (a.u.)
    p_init = 2000.0 # Initial momentum (a.u.)
    mass = 1822.88848*1.00797 # Mass of particle (a.u.), assuming hydrogen

    # Compute the initial velocity
    v_init = p_init / mass  # a.u. for velocity

    # Initial positions (3D system, but keeping everything on x-axis)
    positions = np.array([[x_init, 0.0, 0.0]])  # y, z components are 0

    # Initial velocities
    velocities = np.array([[v_init, 0.0, 0.0]])
    np.savetxt('vfile.txt', velocities, fmt='%.10f') # save to vfile.txt
    temp_tmp = get_temperature(np.array([mass]), velocities) # have to save current temperature since step zero does a mandatory rescaling

    # Prepare FSSH-specific parameters (computed below)
    gradients = None  # Placeholder until computed
    active_surface = 0  # Start on the lower adiabatic state
    td_coeffs = np.array([1.0 + 0j, 0j])  # Start with full population in state 0
    num_TDNAC = False  # Non-adiabatic coupling is computed, not numerical

    # Time step size
    dt = 0.005  # Time step in fs

    # Helper function to compute new gradients, energies, and nonadiabatic couplings
    def compute_quantities(x, A, B, C, D, E0, model_num):
        V1, V2, d12, grad_V1, grad_V2 = compute(x, A, B, C, D, E0, model_num)
        energies = [V1, V2]  # Energies of lower and upper states
        gradients = [np.array([grad_V1, 0, 0]), np.array([grad_V2, 0, 0])] # Gradients of the potential energies
        dcs = [[0, np.array([d12, 0, 0])], [-1 * np.array([d12, 0, 0]), 0]] # Nonadiabatic couplings matrix (antisymmetric)
        return energies, gradients, dcs

    # Initial computation of energies, gradients, and nonadiabatic couplings
    energies, gradients, dcs = compute_quantities(x_init, A, B, C, D, E0, model_num)
    np.savetxt('gfile.txt', gradients[0], fmt='%.10f') # save to gfile.txt

    # Define a new FSSH object for each time step with the updated values
    fssh_simulation = FSSH(
        gradients=gradients, 
        active_surface=active_surface, 
        dcs=dcs, 
        td_coeffs=td_coeffs, 
        num_TDNAC=num_TDNAC, 
        symbols=["H"],              # Assuming Hydrogen (make sure mass above matches)
        positions=positions, 
        nsurf=2, 
        energies=energies, 
        dt=dt, 
        stepnum=8,                  # Picked random number here since we are only doing one step, does not effect calculation
        temperature=temp_tmp,       # Using the initial temperarture so the rescaling does not alter the velocity
        quant_centers=np.array([]), 
        fixed_centers=np.array([]), 
        conv2bohr=False, 
        vel_init=True
    )
    
    # Run one step of dynamics
    fssh_simulation.run()

    # Read updated active surface from active_surface.txt
    with open('active_surface.txt', 'r') as active_surface_file:
        active_surface = int(active_surface_file.readline().strip())

    if (active_surface == 1):
        num_switch = num_switch + 1

# Check if change in ground state probability times number of trajectories gives you the number of swtiches recorded
td_coeffs = np.genfromtxt('tdfile.txt', dtype=complex)
how_many_should_switch = (1.0-np.abs(td_coeffs[0])**2)*ntraj
print("\n")
print(f"             # trajectories that switched = {num_switch}")
print(f" # trajectories that should have switched = {round(how_many_should_switch)}")
print("Note that it is normal for these two quantities to not be the same (~ +/- 10%), \nBUT their agreement should improve as ntraj is increased.")


############################################################################################################
# CHECK VELOCITY RESCALE
# We want to make sure that upon a successful hop that the velcoity is rescaled in order to conserve energy. 
# The first example below is hardcoded for a successful hop, and the second is for a furstrated hop.
############################################################################################################
"""
# SUCCESSFUL HOP
print("\nSUCCESSFUL HOP RESCALE:\n")

from namd.calc_classes import FSSH
from namd.sys_info import get_temperature
    
# Define parameters for Tully Model
A = 10
B = 1.6
C = 0.005
D = 1.0
E0 = 0.05 
model_num = 1

# Initial conditions
x_init = 0  # Initial position (a.u.)
p_init = 20 # Initial momentum (a.u.)
mass = 1822.88848*1.00797 # Mass of particle (a.u.), assuming hydrogen

# Compute the initial velocity
v_init = p_init / mass  # a.u. for velocity

# Initial positions (3D system, but keeping everything on x-axis)
positions = np.array([[x_init, 0.0, 0.0]])  # y, z components are 0

# Initial velocities
velocities = np.array([[v_init, 0.0, 0.0]])
np.savetxt('vfile.txt', velocities, fmt='%.10f') # save to vfile.txt
temp_tmp = get_temperature(np.array([mass]), velocities) # have to save current temperature since step zero does a mandatory rescaling

# Prepare FSSH-specific parameters (computed below)
gradients = None  # Placeholder until computed
active_surface = 0  # Start on the lower adiabatic state
td_coeffs = np.array([1.0 + 0j, 0j])  # Start with full population in state 0
num_TDNAC = False  # Non-adiabatic coupling is computed, not numerical

# Time step size
dt = 0.005  # Time step in fs

# Helper function to compute new gradients, energies, and nonadiabatic couplings
def compute_quantities(x, A, B, C, D, E0, model_num):
    V1, V2, d12, grad_V1, grad_V2 = compute(x, A, B, C, D, E0, model_num)
    energies = [V1, V2]  # Energies of lower and upper states
    gradients = [np.array([grad_V1, 0, 0]), np.array([grad_V2, 0, 0])] # Gradients of the potential energies
    dcs = [[0, np.array([d12, 0, 0])], [-1 * np.array([d12, 0, 0]), 0]] # Nonadiabatic couplings matrix (antisymmetric)
    return energies, gradients, dcs

# Initial computation of energies, gradients, and nonadiabatic couplings
energies, gradients, dcs = compute_quantities(x_init, A, B, C, D, E0, model_num)
np.savetxt('gfile.txt', gradients[0], fmt='%.10f') # save to gfile.txt

print(f"           Initial KE = {(1/2)*mass*v_init**2:0.10f}")
print(f"           Initial PE = {energies[0]:0.10f}")
print(f" Initial Total Energy = {(1/2)*mass*v_init**2+energies[0]:0.10f}")


# Define a new FSSH object for each time step with the updated values
fssh_simulation = FSSH(
    gradients=gradients, 
    active_surface=active_surface, 
    dcs=dcs, 
    td_coeffs=td_coeffs, 
    num_TDNAC=num_TDNAC, 
    symbols=["H"],              # Assuming Hydrogen (make sure mass above matches)
    positions=positions, 
    nsurf=2, 
    energies=energies, 
    dt=dt, 
    stepnum=8,                  # Picked random number here since we are only doing one step, does not effect calculation
    temperature=temp_tmp,       # Using the initial temperarture so the rescaling does not alter the velocity
    quant_centers=np.array([]), 
    fixed_centers=np.array([]), 
    conv2bohr=False, 
    vel_init=True
)

# Run one step of dynamics
fssh_simulation.run()

# FRUSTRATED HOP
print("\nFRUSTRATED HOP RESCALE:\n")

from namd.calc_classes import FSSH
from namd.sys_info import get_temperature

for i in range(2000): # We need to loop over many trajectories so you see (at least a couple) of frustrated hops
    
    # Define parameters for Tully Model
    A = 10
    B = 1.6
    C = 0.005
    D = 1.0
    E0 = 0.05 
    model_num = 1


    # Initial conditions
    x_init = 0  # Initial position (a.u.)
    p_init = 0.2 # Initial momentum (a.u.)
    mass = 1822.88848*1.00797 # Mass of particle (a.u.), assuming hydrogen

    # Compute the initial velocity
    v_init = p_init / mass  # a.u. for velocity

    # Initial positions (3D system, but keeping everything on x-axis)
    positions = np.array([[x_init, 0.0, 0.0]])  # y, z components are 0

    # Initial velocities
    velocities = np.array([[v_init, 0.0, 0.0]])
    np.savetxt('vfile.txt', velocities, fmt='%.10f') # save to vfile.txt
    temp_tmp = get_temperature(np.array([mass]), velocities) # have to save current temperature since step zero does a mandatory rescaling

    # Prepare FSSH-specific parameters (computed below)
    gradients = None  # Placeholder until computed
    active_surface = 0  # Start on the lower adiabatic state
    td_coeffs = np.array([1.0 + 0j, 0j])  # Start with full population in state 0
    num_TDNAC = False  # Non-adiabatic coupling is computed, not numerical

    # Time step size
    dt = 0.005  # Time step in fs

    # Helper function to compute new gradients, energies, and nonadiabatic couplings
    def compute_quantities(x, A, B, C, D, E0, model_num):
        V1, V2, d12, grad_V1, grad_V2 = compute(x, A, B, C, D, E0, model_num)
        energies = [V1, V2]  # Energies of lower and upper states
        gradients = [np.array([grad_V1, 0, 0]), np.array([grad_V2, 0, 0])] # Gradients of the potential energies
        dcs = [[0, np.array([d12, 0, 0])], [-1 * np.array([d12, 0, 0]), 0]] # Nonadiabatic couplings matrix (antisymmetric)
        return energies, gradients, dcs

    # Initial computation of energies, gradients, and nonadiabatic couplings
    energies, gradients, dcs = compute_quantities(x_init, A, B, C, D, E0, model_num)
    np.savetxt('gfile.txt', gradients[0], fmt='%.10f') # save to gfile.txt

    print(f"\n           Initial KE = {(1/2)*mass*v_init**2:0.10f}")
    print(f"           Initial PE = {energies[0]:0.10f}")
    print(f" Initial Total Energy = {(1/2)*mass*v_init**2+energies[0]:0.10f}")

    # Define a new FSSH object for each time step with the updated values
    fssh_simulation = FSSH(
        gradients=gradients, 
        active_surface=active_surface, 
        dcs=dcs, 
        td_coeffs=td_coeffs, 
        num_TDNAC=num_TDNAC, 
        symbols=["H"],              # Assuming Hydrogen (make sure mass above matches)
        positions=positions, 
        nsurf=2, 
        energies=energies, 
        dt=dt, 
        stepnum=8,                  # Picked random number here since we are only doing one step, does not effect calculation
        temperature=temp_tmp,       # Using the initial temperarture so the rescaling does not alter the velocity
        quant_centers=np.array([]), 
        fixed_centers=np.array([]), 
        conv2bohr=False, 
        vel_init=True
    )

    # Run one step of dynamics
    fssh_simulation.run()
    i = i + 1
"""