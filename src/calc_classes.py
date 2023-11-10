# Different calculation classes: AIMD (ground state), Ehrenfest, and FSSH
# Shared attributes are in the Calculation parent class
# Method-specific parameters are in the individual (child) classes

import os
from src.file_io import *
from src.get_surfaces import *
from src.sys_info import *
from src.velocity_init import *
from src.vverlet import *

class Calculation: 
    
    """
    Calculation parent class. All the below attributes are shared amonnst all calculation types.

    Attributes:
        symbols (list): List of length N containing strings fo chemical symbols for the nuceli
        positions (numpy.ndarray): Aarray of shape (N, 3) representing nuclear positions, in bohr
        nsurf (int): Number of PESs the dynamics will be dictated by
        energies (list): List of length nsurf containing the energies of the different surfaces, in hartrees
        dt (float): Time step used to propagate the classical nuclei, in femtoseconds
        stepnum (int): Current step number (0 is the inital time step)
        temperature (float): Temperature used to set the kinetic energy 3/2NkT for inital velocities, in Kelvin
        quant_centers (numpy.ndarray): Array of integers giving indices for which nuclei should be given no mass or velocity
        conv2bohr (boolean): True if we need to convert positions from bohr to Angstrom in final printout, false if we do not need to
    """

    def __init__(self, symbols, positions, nsurf, energies, dt, stepnum, temperature, quant_centers, conv2bohr):
        self.symbols = symbols
        self.positions = positions
        self.nsurf = nsurf
        self.energies = energies
        self.dt = dt
        self.stepnum = stepnum
        self.temperature = temperature
        self.quant_centers = quant_centers
        self.conv2bohr = conv2bohr

class AIMD(Calculation):
    
    """
    AIMD (adiabatic ground state dynamics) class. 

    Attributes:
        delE0 (numpy.ndarray): An array of shape (N, 3) giving the gradient of the ground state energy, in hartree/bohr
    """

    def __init__(self, delE0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n AIMD REQUESTED: classical nuclei will be propagated on the ground state surface")
        self.delE0 = delE0
    
    def run(self):
  
        """
        AIMD run function. Obtains new coordinates for the next time step.
        """

        # Obtain masses of each center and initialize their velocities
        masses = get_masses(self.symbols, self.quant_centers)

        # Construct the gradient the classical nuclei move on
        # In the case of AIMD, this is just the ground state gradient, 
        # but this is not necessarily the case for Ehrenfest or FSSH, where
        # the gradient will have to be calcualted (Ehrenfest) or chosen (FSSH)
        grad = self.delE0

        # Obtain next velocities via velocity Verlet (or initialize them)
        if self.stepnum == 0:
            vels = get_initial_vels(masses, self.positions, self.temperature)
        else: 
            prev_vels = np.genfromtxt('vfile.txt', dtype=float)
            prev_grad = np.genfromtxt('gfile.txt', dtype=float)
            vels = get_next_velocity(masses, prev_vels, prev_grad, grad, self.dt)

        # Get the next coordinates via velocity Verlet
        next_positions = get_next_position(masses, self.positions, vels, grad, self.dt)

        # Print relevant info at this time step
        print_info(masses, self.energies, vels, next_positions, grad, self.quant_centers)

        # Save the velcoity, positions, and gradient for next time step
        save_info(self.symbols, next_positions, vels, grad, new_coeffs, self.conv2bohr)

class Ehrenfest(Calculation):
    
    """
    Ehrenfest dynamics class. 

    Attributes:
        gradients (list of numpy.ndarray): List of arrays of shape (N, 3) giving the gradient of the adiabatic state energies, in hartree/bohr
        dcs (2D list of numpy.ndarray): List (indexed by two indices) where dcs[i][j] give the array of shape (N, 3) for the derivative coupling between adiabats i and j, in 1/bohr
        td_coeffs (numpy.ndarray): Array of complex numbers giving the time-dependent coefficients for each adiabat considered from the previous time step
        num_TDNAC (boolean): True if user is calculating the TD-NAC matrix numerically
    """

    def __init__(self, gradients, dcs, td_coeffs, num_TDNAC, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n EHRENFEST REQUESTED: classical nuclei will be propagated on an average surface ")
        self.gradients = gradients
        self.dcs = dcs
        self.td_coeffs = td_coeffs
        self.num_TDNAC = num_TDNAC
    
    def run(self):
  
        """
        Ehrenfest run function. Obtains new coordinates for the next time step.
        """

        # Obtain masses of each center and initialize their velocities
        masses = get_masses(self.symbols, self.quant_centers)

        # Obtain next velocities via velocity Verlet (or initialize them),
        # calculate the average gradient based on the velocities
        if self.stepnum == 0:
            # Get inital velocities
            vels = get_initial_vels(masses, self.positions, self.temperature)
            # Obtain TD-NAC matrix
            Tmat = get_T_matrix(vels, self.dcs, self.dt, False) # Because we are on t = 0, we can't compute numerical TD-NAC yet, so hardcode False
            # Integrate the TDSE to get the new coefficients for each adiabat
            new_coeffs = get_new_coeffs(self.td_coeffs, self.dt, self.energies, Tmat)
            # Calculate the new average surface
            grad = get_ehrenfest_grad(new_coeffs, self.energies, self.gradients, self.dcs)
        else: 
            # Get previous velocities and gradient of previous average surface
            prev_vels = np.genfromtxt('vfile.txt', dtype=float)
            prev_grad = np.genfromtxt('gfile.txt', dtype=float)
            # Obtain TD-NAC matrix
            Tmat = get_T_matrix(vels, self.dcs, self.dt, self.num_TDNAC)
            # Integrate the TDSE to get the new coefficients for each adiabat
            new_coeffs = get_new_coeffs(self.td_coeffs, self.dt, self.energies, Tmat)
            # Calculate the new average surface
            grad = get_ehrenfest_grad(new_coeffs, self.energies, self.gradients, self.dcs)
            vels = get_next_velocity(masses, prev_vels, prev_grad, grad, self.dt)

        # Get the next coordinates via velocity Verlet
        next_positions = get_next_position(masses, self.positions, vels, grad, self.dt)

        # Print relevant info at this time step
        print_info(masses, self.energies, vels, next_positions, grad, self.quant_centers)

        # Save the velcoity, positions, and gradient for next time step
        save_info(self.symbols, next_positions, vels, grad, new_coeffs, self.conv2bohr)
