#!/usr/bin/python

import domain
import potential
import state

if __name__ == '__main__':
    
    
    print "test MD"
    
    n=10
    N=n**3
    
    rho = 1
    
    mu = 0.0
    nsig = 0.5
    
    
    
    print rho
    
    test_domain = domain.BaseDomain()
    test_potential = potential.LennardJones()
    test_pos_init_lattice = state.PosLatticeInitNRho(N, rho)
    test_vel_init = state.VelNormDistInit(mu,nsig)
    
    
    
    
    
    test_state = state.BaseMDState(domain = test_domain,
                                   potential = test_potential, 
                                   particle_pos_init = test_pos_init_lattice, 
                                   particle_vel_init = test_vel_init,
                                   N = N)
    
    
    test_state.frame_plot()
    
    
    
    
    
    
    
    
