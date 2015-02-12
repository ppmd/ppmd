#!/usr/bin/python

import domain
import potential
import state

if __name__ == '__main__':
    
    
    print "Wip"
    
    N=64
    
    test_domain = domain.BaseDomain()
    test_potential = potential.LennardJones()
    
    test_init_lattice = state.LatticeInitNRho(N, 1.)
    
    test_state = state.BaseMDState(test_domain, test_potential, test_init_lattice, N)

    print test_state.positions()
    
    test_state.frame_plot()
    
    
    
    
    
    
    
    
