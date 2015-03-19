#!/usr/bin/python

import domain
import potential
import state
import numpy as np
import math
import method
import data
import time

if __name__ == '__main__':
    
    #1000 particles in a box
    test_1000 = True
    
    #2 particles bouncing agasint each other.
    test_2_bounce = False
    
    #plot as computing + at end?
    plotting = True
    
    #log energy?
    logging = True
    
    if (test_1000):
        n=10
        N=n**3
        rho = 3.2
        mu = 0.0
        nsig = 2.5
        
        #Initialise basci domain
        test_domain = domain.BaseDomain()
        
        #Initialise LJ potential
        test_potential = potential.LennardJonesShifted(sigma=1.0,epsilon=1.0)    
        
        #Place N particles in a lattice with given density.
        test_pos_init = state.PosInitLatticeNRho(N, rho)
        
        #Normally distributed velocities.
        test_vel_init = state.VelInitNormDist(mu,nsig)
        
        #Initialise masses, in this case sets all to 1.0.
        test_mass_init = state.MassInitTwoAlternating(m1 = 1.0, m2 = 1.0)
        
    if (test_2_bounce):
        N=2
        
        #See above
        test_domain = domain.BaseDomain()
        test_potential = potential.LennardJonesShifted(sigma=1.0,epsilon=1.0)
        
        #Initialise two particles on an axis a set distance apart.
        test_pos_init = state.PosInitTwoParticlesInABox(rx = 0.5, extent = np.array([7., 7., 7.]), axis = np.array([1,0,0]))
        
        #Give first two particles specific velocities
        test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.]))
        
        #Set alternating masses for particles.
        test_mass_init = state.MassInitTwoAlternating(m1 = 1.0, m2 = 10.0)
        
    
    
    #Create state class from above initialisations.
    test_state = state.BaseMDState(domain = test_domain,
                                   potential = test_potential, 
                                   particle_pos_init = test_pos_init, 
                                   particle_vel_init = test_vel_init,
                                   particle_mass_init = test_mass_init,
                                   N = N,
                                   )
    
    
    #Create an integrator for above state class.
    test_integrator = method.VelocityVerlet(state = test_state, USE_C = True, USE_LOGGING = logging, USE_PLOTTING = plotting)
    
    #integrate.
    start = time.clock()
    energy_data = test_integrator.integrate(dt = 0.0001, T = 0.1)
    end = time.clock()
    print "Rough time taken:", end - start,"s"
    
    
    
    #If logging was enabled, plot data.
    if (logging):
        energy_data.plot()
    
    
    
    a=input("PRESS ENTER TO CONTINUE.\n")
    
    
    
    
