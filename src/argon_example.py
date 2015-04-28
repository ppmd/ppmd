#!/usr/bin/python

import domain
import potential
import state
import numpy as np
import math
import method
import data
import loop
import subprocess
import os

if __name__ == '__main__':

    #plot as computing + at end?
    plotting = False
    
    #log energy?
    logging = False
    
    #Enbale debug flags?
    debug = True   
    
    
    
    #particle properties
    N       = 100
    mass    = 39.948        #g/mol is this the correct mass?
    rho     = N/(17.4**3)
    T       = 85.0
    
    #potential properties
    epsilon = 0.9661
    sigma   = 3.405
    cutoff  = 8.5
    

    domain                  = domain.BaseDomain()
    potential               = potential.LennardJones(sigma,epsilon,cutoff)
    
    initial_position_config = state.PosInitDLPOLYConfig('TEST7/CONFIG')
    initial_velocity_config = state.VelInitDLPOLYConfig('TEST7/CONFIG')
    
    
    
    intial_mass_config      = state.MassInitIdentical(mass)
    
    
    
    
    test_state = state.BaseMDState(domain = domain,
                                   potential = potential, 
                                   particle_pos_init = initial_position_config, 
                                   particle_vel_init = initial_velocity_config,
                                   particle_mass_init = intial_mass_config,
                                   N = N,
                                   DEBUG = debug
                                   )
    
    
    
    
    
    
    
    
    #plotting handle
    if (plotting):
        plothandle = data.DrawParticles(interval = 2)
    else:
        plothandle = None
    
    #energy handle
    if (logging):
        energyhandle = data.BasicEnergyStore()
    else:
        energyhandle = None
    
    
    
    
    
    
    integrator = method.VelocityVerletAnderson(state = test_state, USE_C = True, plot_handle = plothandle, energy_handle = energyhandle, writexyz = False, DEBUG = debug)
    
    
    #control file seems to compute 16000 iterations at dt =10^-3
    dt=10**-3
    T=16000*dt
    integrator.integrate(dt = dt, T = T, timer=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if (logging):
        energyhandle.plot()
    
    
    a=input("PRESS ENTER TO CONTINUE.\n")
    
