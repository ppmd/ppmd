#!/usr/bin/python

import domain
import potential
import state

if __name__ == '__main__':
    
    
    print "Wip"
    test_domain = domain.BaseDomain()
    test_potential = potential.LennardJones()
    test_state = state.BaseMDState(test_domain, test_potential, 1)
    tmp=test_state.velocities()
    
    
    print test_state.energy_kenetic()
    
    
    
    
    
    
    
    
    
    
