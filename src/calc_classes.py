# Different calculation classes: AIMD (ground state), Ehrenfest, and FSSH
# Shared parameters are in the Calculation parent class
# Method-specific parameters are in the individual (child) classes

class Calculation: 
    
    # Calculation Parameters 
    # ncen: the number of basis function centers present in the calculation
    #       (if no ghost atoms are present this equals the number of atoms)
    # nsurf: the number of PESs the dynamics will be dictated by
    # dt: time step in femtoseconds

    def __init__(self, ncen, nsurf, dt):
        self.ncen = ncen
        self.nsteps = nsteps
        self.dt = dt

class AIMD(Calculation):
    
    # AIMD (Ground State Dynamics) Parameters
    # delE0: ncen-by-3 numpy array of the ground state gradient on each center 
    
    def __init__(self, delE0, ncen, nsurf=1, dt=0.06):
        super().__init__(ncen=ncen, dt=dt)
        self.delE0 = delE0

