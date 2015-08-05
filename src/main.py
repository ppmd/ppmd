#!/usr/bin/python

import domain
import potential
import state
import numpy as np
import method
import data


if __name__ == '__main__':
    
    x = 1
    y = 0
    while x == 0:
        y += 1

    # 1000 particles in a box
    test_1000 = True
    
    # 2 particles bouncing agasint each other.
    test_2_bounce = True
    
    # 1 1 particle
    t_1_particle = False

    # plot as computing + at end?
    plotting = True
    
    # log energy?
    logging = False
    
    # Enbale debug flags?
    debug = True

    MPI_HANDLE = data.MDMPI()
    
    if test_1000:
        # n=25 reasonable size
        n = 10
        N = n**3
        # n=860
        rho = 0.05
        mu = 0.0
        nsig = 5.0
        
        # Initialise basic domain
        test_domain = domain.BaseDomainHalo(nt=N, periods = (False,False,False), mpi_handle= MPI_HANDLE)
        
        # Initialise LJ potential
        test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)    
        
        # Place n particles in a lattice with given density.
        # test_pos_init = state.PosInitLatticeNRho(n, rho, None)
        test_pos_init = state.PosInitLatticeNRhoRand(N,rho,0.,None)
        
        # Normally distributed velocities.
        test_vel_init = state.VelInitNormDist(mu,nsig)
        
        # Initialise masses, in this case sets all to 1.0.
        test_mass_init = state.MassInitIdentical(1.)
        # test_mass_init = state.MassInitTwoAlternating(200., 1.)
        
        
        
    if test_2_bounce:
        N = 2
        
        # See above
        test_domain = domain.BaseDomainHalo(nt=N, periods = (True,True,True), mpi_handle= MPI_HANDLE)
        test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)
        
        print test_potential.rc

        # Initialise two particles on an axis a set distance apart.
        test_pos_init = state.PosInitTwoParticlesInABox(rx = 0.4, extent = np.array([6., 6., 6.]), axis = np.array([1,0,0]))
        
        # Give first two particles specific velocities
        test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.]))
        
        # Set alternating masses for particles.
        
        test_mass_init = state.MassInitTwoAlternating(5., 5.)
        
    if t_1_particle:
        
        N = 1
        
        # See above
        test_domain = domain.BaseDomainHalo(nt=N, mpi_handle= MPI_HANDLE)
        test_potential = potential.NULL(rc = 0.01)
        
        print test_potential.rc

        # Initialise two particles on an axis a set distance apart.
        test_pos_init = state.PosInitOneParticleInABox(r = np.array([0., 0., 0.]), extent = np.array([0.2, 0.2, 0.2]))
        
        # Give first two particles specific velocities
        test_vel_init = state.VelInitOneParticleInABox(vx = np.array([5., 0., 0.]))
        
        # Set alternating masses for particles.
        
        test_mass_init = state.MassInitIdentical(5.)    

    # Create state class from above initialisations.
    test_state = state.BaseMDStateHalo(domain=test_domain,
                                       potential=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N,
                                       DEBUG=debug
                                       )
    
    # plotting handle
    if plotting:
        plothandle = data.DrawParticles(5, MPI_HANDLE)
    else:
        plothandle = None

    # Create VAF method.
    test_vaf_method = None
    # test_vaf_method = method.VelocityAutoCorrelation(state = test_state, DEBUG = debug)
    

    test_xyz = method.WriteTrajectoryXYZ(state=test_state)
    energyhandle = data.EnergyStore(state=test_state)

    schedule = method.Schedule([1000, 1], [test_xyz.write, energyhandle.update])

    # Create an integrator for above state class.
    test_integrator = method.VelocityVerletAnderson(state = test_state, plot_handle = plothandle, energy_handle = None, writexyz = False, VAF_handle = test_vaf_method, DEBUG = debug, mpi_handle = MPI_HANDLE, schedule=schedule)
    #test_integrator = method.VelocityVerletBox(state = test_state, plot_handle = plothandle, energy_handle = energyhandle, writexyz = False, vaf_handle= test_vaf_method, DEBUG = debug, mpi_handle= MPI_HANDLE, schedule=schedule)
    


    
    #create G(r) method.
    #test_gr_method = method.RadialDistributionPeriodicNVE(state = test_state, rsteps = 200, DEBUG = debug)
    

    '''
    for ix in range(2):
        test_state.forces_update()
    '''
    
    
    ###########################################################
    



    
    test_integrator.integrate(dt = 0.00001, t= 0.05, timer=True)

    #test_integrator.integrate_thermostat(dt = 0.0001, t= 0.5, temp=0.01, nu=2.5, timer=True)
    
    #test_integrator.integrate(dt = 0.0001, t= 2.0, timer=True)
    #
    #test_integrator.integrate(dt = 0.0001, T = 0.1, timer=True)
    #test_gr_method.evaluate(timer=True)
    #test_integrator.integrate(dt = 0.0001, T = 0.1, timer=True)
    #test_gr_method.evaluate(timer=True)



    if (MPI_HANDLE is None or MPI_HANDLE.rank ==0):
        print "Total time in halo exchange:", test_domain.halos._time
        print "Time in forces_update:", test_state._time_prof
    
    ###########################################################
    
    

    #If logging was enabled, plot data.
    if (logging):
        energyhandle.plot()
        
        
    #test_gr_method.plot()
    #test_gr_method.raw_write()
    #test_vaf_method.plot()
    
    
    
    if (MPI_HANDLE is None or MPI_HANDLE.rank ==0):
        try:
            a=input("PRESS ENTER TO CONTINUE.\n")
            #pass
        except:
            pass
    #MPI_HANDLE.barrier()

    
    
    
    
    
    
    
    
    
    
