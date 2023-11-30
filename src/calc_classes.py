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
            print("\n INITIAL VELOCITIES (in bohr/(au of time))")
            print(vels)
        else: 
            prev_vels = np.genfromtxt('vfile.txt', dtype=float)
            prev_grad = np.genfromtxt('gfile.txt', dtype=float)
            vels = get_next_velocity(masses, prev_vels, prev_grad, grad, self.dt)

        # Get the next coordinates via velocity Verlet
        next_positions = get_next_position(masses, self.positions, vels, grad, self.dt)

        # Print relevant info at this time step
        print_info(masses, self.energies[0], vels, next_positions, grad, self.quant_centers)

        # Save the velcoity, positions, and gradient for next time step
        save_info(self.symbols, next_positions, vels, grad, np.array([]), self.conv2bohr)

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
        Ehrenfest run function. Obtains new coordinates for the next time step propagated on an average surface.
        """

        # Obtain masses of each center and initialize their velocities
        masses = get_masses(self.symbols, self.quant_centers)

        # Obtain next velocities via velocity Verlet (or initialize them), calculate the average gradient based on the velocities
        if self.stepnum == 0:
            # Get inital velocities
            vels = get_initial_vels(masses, self.positions, self.temperature)
            print("\n INITIAL VELOCITIES (in bohr/(au of time))")
            print(vels)
            # Get gradient
            grad = get_ehrenfest_grad(self.td_coeffs, self.energies, self.gradients, self.dcs)
            new_coeffs = self.td_coeffs # only for pe calculation below
        else: 
            # Get previous velocities and gradient of previous average surface
            prev_vels = np.genfromtxt('vfile.txt', dtype=float)
            prev_grad = np.genfromtxt('gfile.txt', dtype=float)
            # Obtain TD-NAC matrix
            Tmat = get_T_matrix(prev_vels, self.dcs, self.dt, self.num_TDNAC)
            # Integrate the TDSE to get the new coefficients for each adiabat
            new_coeffs, time_points, y_values = get_new_coeffs(self.td_coeffs, self.dt, self.energies, Tmat)
            # Calculate the new average surface
            grad = get_ehrenfest_grad(new_coeffs, self.energies, self.gradients, self.dcs)
            vels = get_next_velocity(masses, prev_vels, prev_grad, grad, self.dt)

        # Get the next coordinates via velocity Verlet
        next_positions = get_next_position(masses, self.positions, vels, grad, self.dt)

        # Print relevant info at this time step
        pe = np.sum(np.abs(new_coeffs[i])**2 * self.energies[i] for i in range(len(new_coeffs)))
        print_info(masses, pe, vels, next_positions, grad, self.quant_centers)

        # Save the velcoity, positions, and gradient for next time step
        save_info(self.symbols, next_positions, vels, grad, new_coeffs, self.conv2bohr)

class FSSH(Calculation):
    
    """
    FSSH dynamics class. 

    Attributes:
        gradients (list of numpy.ndarray): List of arrays of shape (N, 3) giving the gradient of the adiabatic state energies, in hartree/bohr
        active_surface (int): Integer indicating which adiabatic state is currently active 
        dcs (2D list of numpy.ndarray): List (indexed by two indices) where dcs[i][j] give the array of shape (N, 3) for the derivative coupling between adiabats i and j, in 1/bohr
        td_coeffs (numpy.ndarray): Array of complex numbers giving the time-dependent coefficients for each adiabat considered from the previous time step
        num_TDNAC (boolean): True if user is calculating the TD-NAC matrix numerically
    """

    def __init__(self, gradients, active_surface, dcs, td_coeffs, num_TDNAC, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n FSSH REQUESTED: classical nuclei will be on one adiabatic state at a time, with switches according to the FSSH algroithm ")
        self.gradients = gradients
        self.active_surface = active_surface
        self.dcs = dcs
        self.td_coeffs = td_coeffs
        self.num_TDNAC = num_TDNAC
    
    def run(self):
  
        """
        FSSH run function. Obtains new coordinates for the next time step, determines if we swtich to a new adiabatic state.
        """

        # In FSSH, we evolve R and v in the same way as AIMD on the current active surface. 
        # After that, we determine if a hop occurs. If it does, we adjust the velocity accordingly. 
        # If the hop was successful (ie not frustrated), the new active surface will take effect next step

        # Obtain masses of each center and initialize their velocities
        masses = get_masses(self.symbols, self.quant_centers)

        # Get the current ative surface
        grad = self.gradients[self.active_surface]

        # Obtain next velocities via velocity Verlet (or initialize them)
        if self.stepnum == 0:
            vels = get_initial_vels(masses, self.positions, self.temperature)
            print("\n INITIAL VELOCITIES (in bohr/(au of time))")
            print(vels)
        else: 
            prev_vels = np.genfromtxt('vfile.txt', dtype=float)
            prev_grad = np.genfromtxt('gfile.txt', dtype=float)
            vels = get_next_velocity(masses, prev_vels, prev_grad, grad, self.dt)

        # Get the next coordinates via velocity Verlet
        next_positions = get_next_position(masses, self.positions, vels, grad, self.dt)

        # We have evolved R and v on this surface. We now determine if a hop will occur. 

        print_coeffs = self.td_coeffs # default coeffs to print 
        if self.stepnum != 0: # inital step should be equivalent to an AIMD step

            # Obtain T matrix (using previous velocities, it is not the new velcoities that determine the Tmat)
            Tmat = get_T_matrix(prev_vels, self.dcs, self.dt, self.num_TDNAC)
        
            # Integrate the TDSE to get the new coefficients for each adiabat
            new_coeffs, time_points, y_values = get_new_coeffs(self.td_coeffs, self.dt, self.energies, Tmat)
            print_coeffs = new_coeffs
            
            # Determine if hop occurs 
            hop_check = check_hop(time_points, y_values, self.active_surface, Tmat)

            # Rescale velocities if a hop occured
            if hop_check != -1:
                print(f"\n HOP FROM STATE {self.active_surface} TO STATE {hop_check} ATTEMPTED...")
                nacv = self.dcs[self.active_surface][hop_check]
                ediff = self.energies[hop_check] - self.energies[self.active_surface]
                vels, frustrated = rescale_vels(vels, nacv, masses, ediff) # vels should be rewritten in case of frustrated or non-frustrated hop
                if not frustrated: # however, we only change the active surface if the hop was not frustrated
                    # Set gradient to the new active surface
                    grad = self.gradients[hop_check]
                self.active_surface = hop_check # set active surface to the one we hopped to

        # Print relevant info at this time step
        print_info(masses, self.energies[self.active_surface], vels, next_positions, grad, self.quant_centers)

        # Save the velocity, positions, and gradient for next time step
        save_info(self.symbols, next_positions, vels, grad, print_coeffs, self.conv2bohr)