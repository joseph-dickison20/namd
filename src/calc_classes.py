# Different calculation classes: AIMD (ground state), Ehrenfest, and FSSH
# Shared attributes are in the Calculation parent class
# Method-specific parameters are in the individual (child) classes

import os
from src.file_io import *
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

        # Print relevant info at this time step
        aimd_print_info(masses, self.energies, vels, self.positions, grad, self.quant_centers)

        # Get the next coordinates via velocity Verlet
        next_positions = get_next_position(masses, self.positions, vels, grad, self.dt)

        # Save the velcoity, positions, and gradient for next time step
        save_info(self.symbols, next_positions, vels, grad, self.conv2bohr)
