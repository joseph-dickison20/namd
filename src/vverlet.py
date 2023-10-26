# Velocity Verlet routine
# Note that we are only doing one time step at a time, 
# so there is no loop needed.

import numpy as np

fs2au = 41.3413733365 # there are this mnay au of time in one femtosecond, the unit dt is given in

def get_accel(masses, grad):
    # a = F/m = -delE/m
    # Suppress the divide by zero error, but then turn it back on
    original_settings = np.geterr()
    np.seterr(divide='ignore', invalid='ignore')
    # Set accels where mass is zero to zero, that way original position of x_prev is preserved
    accels = np.where(masses[:, np.newaxis] != 0, -1*grad / masses[:, np.newaxis], 0)
    np.seterr(**original_settings) # Restore the original error settings
    return accels

def get_next_velocity(masses, prev_vels, prev_grad, grad, dt):
    # v_new = v_prev + 0.5*(a_prev+a_new)*dt, a = F/m
    dt *= fs2au # convert fs to au
    accels = get_accel(masses, grad)
    accels_prev = get_accel(masses, prev_grad)
    return prev_vels + 0.5*(accels+accels_prev)*dt

def get_next_position(masses, positions, vels, grad, dt):
    # x_new = x_old + v*dt + 0.5*a*dt^2, a = F/m
    dt *= fs2au # convert fs to au
    accels = get_accel(masses, grad)
    return positions + vels*dt + 0.5*accels*dt**2
    