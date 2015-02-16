#!/usr/bin/python

import domain
import potential
import state

if __name__ == '__main__':
    
    
    print "test MD"
    
    N=27
    rho = 1.
    
    test_domain = domain.BaseDomain()
    test_potential = potential.LennardJones()
    
    test_init_lattice = state.LatticeInitNRho(N, rho)
    
    test_state = state.BaseMDState(test_domain, test_potential, test_init_lattice, N)
    
    
    
    
    
    #test_state.frame_plot()
    
    
    
    
    
    
    
    
