# Different calculation classes: AIMD (ground state), Ehrenfest, and FSSH
# Shared attributes are in the Calculation parent class
# Method-specific parameters are in the individual (child) classes

from src.sys_info import *
from src.velocity_init import *

class Calculation: 
    
    """
    Calculation parent class. All the below attributes are shared amonnst all calculation types.

    Attributes:
        symbols (list): A list of length N containing strings fo chemical symbols for the nuceli
        positions (numpy.ndarray): An array of shape (N, 3) representing nuclear positions, in bohr
        nsurf (int): the number of PESs the dynamics will be dictated by
        dt (float): the time step used to propagate the classical nuclei, in femtoseconds
        temperature (float): temperature used to set the kinetic energy 3/2NkT for inital velocities, in Kelvin
        quant_centers (numpy.ndarray): An array of integers giving indices for which nuclei should be given no mass or velocity
    """

    def __init__(self, symbols, positions, nsurf=1, dt=0.06, temperature=298.15, quant_centers=np.array([])):
        self.symbols = symbols
        self.positions = positions
        self.nsurf = nsurf
        self.dt = dt
        self.temperature = temperature
        self.quant_centers = quant_centers

class AIMD(Calculation):
    
    """
    AIMD (adiabatic ground state dynamics) class. 

    Attributes:
        delE0 (numpy.ndarray): An array of shape (N, 3) giving the gradient of the ground state energy, in hartree/bohr
    """

    def __init__(self, delE0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delE0 = delE0
    
    def run(self):
        """
        AIMD run function. Obtains new coordinates for the next time step.
        """
        # Obtain masses of each cneter and initialize their velocities
        masses = get_masses(self.symbols, self.quant_centers)
        vel_inits = get_initial_vels(masses, self.positions, self.temperature)
        # Perform velocity Verlet toget the next coordinates
        # Put VVerlet code here
