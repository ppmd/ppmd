#!/usr/bin/python

import domain
import potential
import state
import numpy as np
import math

if __name__ == '__main__':
    
    
    print "test MD"
    
    #n=10
    #N=n**3
    #rho = 3.2
    
    N=2
    
    
    
    mu = 0.0
    nsig = 2.5
    
    
    
    
    
    test_domain = domain.BaseDomain()
    test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)
    
    #test_pos_init = state.PosInitLatticeNRho(N, rho)
    #test_vel_init = state.VelInitNormDist(mu,nsig)
    
    test_pos_init = state.PosInitTwoParticlesInABox(rx = 0.5, extent = np.array([7., 7., 7.]), axis = np.array([1,0,0]))
    test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([1., 0., 0.]), vy = np.array([-1., 0., 0.]))
    
    r=0.5
    #test_pos_init = state.PosInitTwoParticlesInABox(rx = r, extent = np.array([7., 7., 7.]), axis = np.array([1,0,0]))
    #test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([0., 1./(math.sqrt(2*r)), 0.]), vy = np.array([0., -1./(math.sqrt(2*r)), 0.]))
    
    test_state = state.BaseMDState(domain = test_domain,
                                   potential = test_potential, 
                                   particle_pos_init = test_pos_init, 
                                   particle_vel_init = test_vel_init,
                                   N = N,
                                   dt = 0.00001,
                                   T =  0.05
                                   )
    
    
    test_state.frame_plot_energy()
    
    
    
    a=input("PRESS ENTER TO CONTINUE.\n")
    
    
    
    
