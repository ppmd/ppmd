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
import getopt
import sys


if __name__ == '__main__':

    MPI_HANDLE = data.MDMPI()

    #plot as computing + at end?
    plotting = False
    
    #log energy?
    logging = False
    
    #Enbale debug flags?
    debug = True
    
    #print things?
    verbose = False
    
    
    
    #particle properties
    N       = 10**3
    I       = 5000
    
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "N:I:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    for o, a in opts:
        if o == "-N":
            N=int(a)
        elif o == "-I":
            I=int(a)                        
        else:
            assert False, "unhandled option"    
    
    
    
    mass    = 39.948        #g/mol is this the correct mass?
    
    #potential properties
    epsilon = 0.9661
    sigma   = 3.405
    cutoff  = 7.5
    
    
    domain                  = domain.BaseDomainHalo(MPI_handle = MPI_HANDLE)
    potential               = potential.LennardJonesCounter(epsilon,sigma,cutoff)
    
    #initial_position_config = state.PosInitDLPOLYConfig('TEST7_CUSTOM/CONFIG')
    #initial_velocity_config = state.VelInitDLPOLYConfig('TEST7_CUSTOM/CONFIG')
    
    #initial_position_config = state.PosInitDLPOLYConfig('../util/REVCON')
    #initial_velocity_config = state.VelInitDLPOLYConfig('../util/REVCON')    
    
    initial_position_config = state.PosInitDLPOLYConfig('../util/CONFIG')
    initial_velocity_config = state.VelInitDLPOLYConfig('../util/CONFIG')     
    
    
    intial_mass_config      = state.MassInitIdentical(mass)
    
    
    
    
    test_state = state.BaseMDStateHalo(domain = domain,
                                       potential = potential, 
                                       particle_pos_init = initial_position_config, 
                                       particle_vel_init = initial_velocity_config,
                                       particle_mass_init = intial_mass_config,
                                       N = N,
                                       DEBUG = debug,
                                       MPI_handle = MPI_HANDLE
                                       )
    
    
    
    
    
    
    
    
    #plotting handle
    if (plotting):
        plothandle = data.DrawParticles(interval = 25, MPI_handle = MPI_HANDLE)
    else:
        plothandle = None
    
    #energy handle
    if (logging):
        energyhandle = data.BasicEnergyStore(MPI_handle = MPI_HANDLE)
    else:
        energyhandle = None
    
    
    
    
    
    
    integrator = method.VelocityVerletAnderson(state = test_state, USE_C = True, plot_handle = plothandle, energy_handle = energyhandle, writexyz = False, DEBUG = debug, MPI_handle = MPI_HANDLE)
    
    
    #control file seems to compute 16000 iterations at dt =10^-3, 1000 to equbrilate then 15k for averaging?
    dt=10**-3
    T=I*dt
    integrator.integrate(dt = dt, T = T, timer=True)
    
    if (verbose):
        print test_state.positions[0,::]
        print test_state.velocities[0,::]    
        print test_state.forces[0,::]  
    
    if (verbose==True and MPI_HANDLE.rank==0):
        print "Total time in halo exchange:", domain.halos._time
        print "Time in forces_update:", test_state._time    
    
    
    
    
    
    print "COUNT", potential._counter.Dat[0], "OUTER COUNT", potential._counter_outer.Dat[0]
    
    
    
    
    if (logging):
        energyhandle.plot()
    
    
    #a=input("PRESS ENTER TO CONTINUE.\n")
    
