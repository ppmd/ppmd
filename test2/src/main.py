#!/usr/bin/python

import domain
import potential
import state

if __name__ == '__main__':
    
    
    print "test MD"
    
    N=10**3
    rho = 3
    dev = 0.0
    
    print rho
    
    test_domain = domain.BaseDomain()
    test_potential = potential.LennardJonesShifted()
    
    test_init_lattice = state.LatticeInitNRhoRand(N, rho, dev)
    
    test_state = state.BaseMDState(test_domain, test_potential, test_init_lattice, N)
    
    
    test_state.frame_plot()
    
    
    
    
    
    
    
    
