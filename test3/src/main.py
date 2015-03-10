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
    
    
    test_1000 = False
    test_2_bounce = True
    
    if (test_1000):
        n=10
        N=n**3
        rho = 3.2
        mu = 0.0
        nsig = 2.5
        
        test_domain = domain.BaseDomain()
        test_potential = potential.LennardJonesShifted(sigma=1.0,epsilon=1.0)    
    
        test_pos_init = state.PosInitLatticeNRho(N, rho)
        test_vel_init = state.VelInitNormDist(mu,nsig)    
    
    if (test_2_bounce):
        N=2
        
        test_domain = domain.BaseDomain()
        test_potential = potential.LennardJonesShifted(sigma=1.0,epsilon=1.0)
        
        test_pos_init = state.PosInitTwoParticlesInABox(rx = 0.5, extent = np.array([7., 7., 7.]), axis = np.array([1,0,0]))
        test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.]))
        
        

    #r=0.5
    #test_pos_init = state.PosInitTwoParticlesInABox(rx = r, extent = np.array([7., 7., 7.]), axis = np.array([1,0,0]))
    #test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([0., 1./(math.sqrt(2*r)), 0.]), vy = np.array([0., -1./(math.sqrt(2*r)), 0.]))
    
    
    
    test_state = state.BaseMDState(domain = test_domain,
                                   potential = test_potential, 
                                   particle_pos_init = test_pos_init, 
                                   particle_vel_init = test_vel_init,
                                   N = N,
                                   )
    
    
    test_integrator = method.VelocityVerlet(state = test_state, USE_C = True)
    
    start = time.clock()
    energy_data = test_integrator.integrate(dt = 0.00001, T = 0.1)
    end = time.clock()
    print "Rough time taken:", end - start,"s"
    
    
    
    
    
    energy_data.plot()
    
    
    
    a=input("PRESS ENTER TO CONTINUE.\n")
    
    
    
    
