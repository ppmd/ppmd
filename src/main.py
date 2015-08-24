#!/usr/bin/python

import runtime
# debug level
runtime.DEBUG.level = 1
#verbosity level
runtime.VERBOSE.level = 3
#timer level
runtime.TIMER.level = 3

#cuda on/off
runtime.CUDA_ENABLED.flag = True


import domain
import potential
import state
import numpy as np
import method
import data
import gpucuda

import particle

if __name__ == '__main__':


    x = 1
    y = 0
    while x == 0:
        y += 1

    # 1000 particles in a box
    test_1000 = True
    
    # 2 particles bouncing agasint each other.
    test_2_bounce = False
    
    # 1 1 particle
    t_1_particle = False

    # plot as computing + at end?
    plotting = False
    
    # log energy?
    logging = False

    # Write XYZ?
    writing = True

    t=0.001
    dt=0.0001

    # check gpucuda Module initalised correctly.
    '''
    if gpucuda.INIT_STATUS():
        a_N = 1000

        b = particle.Dat(initial_value=list(np.linspace(0,10,a_N-1))); b.add_cuda_dat() ;b.copy_to_cuda_dat()
        c = particle.Dat(initial_value=list(np.linspace(0,10,a_N-1))); c.add_cuda_dat() ;c.copy_to_cuda_dat()
        a = particle.Dat(initial_value=list(np.linspace(0,10,a_N-1))); a.add_cuda_dat() ;a.copy_to_cuda_dat()

        print a[-1], a.dat[0:10:]

        a.dat[0:a_N:] = 0.0
        print a[0:10:]

        _aebpc = gpucuda.aebpc(a,b,c).execute()

        a.copy_from_cuda_dat()
        print a[0:10:]
        print a[-1]
    else:
        print runtime.MPI_HANDLE.rank, "gpucuda not init", gpucuda.INIT_STATUS()
    '''



    if test_1000:
        # n=25 reasonable size
        n = 25
        N = n**3
        # n=860
        rho = 1.5
        mu = 0.0
        nsig = 5.0
        
        # Initialise basic domain
        test_domain = domain.BaseDomain(nt=N)

        # Initialise LJ potential
        test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)    
        
        # Place n particles in a lattice with given density.
        # test_pos_init = state.PosInitLatticeNRho(n, rho, None)
        test_pos_init = state.PosInitLatticeNRhoRand(N,rho,0.,None)
        
        # Normally distributed velocities.
        test_vel_init = state.VelInitNormDist(mu,nsig)
        # test_vel_init = state.VelInitPosBased()
        
        # Initialise masses, in this case sets all to 1.0.
        test_mass_init = state.MassInitIdentical(1.)
        # test_mass_init = state.MassInitTwoAlternating(100., 100.)
        
        
        
    if test_2_bounce:
        N = 2
        dt=0.00001
        
        # See above
        test_domain = domain.BaseDomain(nt=N)
        test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)


        # Initialise two particles on an axis a set distance apart.
        test_pos_init = state.PosInitTwoParticlesInABox(rx = 0.3, extent = np.array([10., 10., 10.]), axis = np.array([1,1,0]))

        # Give first two particles specific velocities
        test_vel_init = state.VelInitTwoParticlesInABox(vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.]))

        # Set alternating masses for particles.
        
        test_mass_init = state.MassInitTwoAlternating(10., 5.)
        
    if t_1_particle:
        
        N = 1
        
        # See above
        test_domain = domain.BaseDomainHalo(nt=N)
        test_potential = potential.NULL(rc = 0.01)
        
        print test_potential.rc

        # Initialise two particles on an axis a set distance apart.
        test_pos_init = state.PosInitOneParticleInABox(r = np.array([0., 0., 0.]), extent = np.array([0.2, 0.2, 0.2]))
        
        # Give first two particles specific velocities
        test_vel_init = state.VelInitOneParticleInABox(vx = np.array([5., 0., 0.]))
        
        # Set alternating masses for particles.
        
        test_mass_init = state.MassInitIdentical(5.)    


    # Create state class from above initialisations.
    test_state = state.BaseMDStateHalo(domain_in=test_domain,
                                       potential_in=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N
                                       )

    # plotting handle
    if plotting:
        plothandle = data.DrawParticles(state=test_state)
        plotsteps = 1000
        plotfn = plothandle.draw
    else:
        plothandle = None
        plotsteps = 0
        plotfn = None

    # xyz writing handle
    if writing:
        test_xyz = method.WriteTrajectoryXYZ(state=test_state)
        test_xyz_steps = 20
        writefn = test_xyz.write
    else:
        test_xyz_steps = 0
        writefn = None

    # xyz writing handle
    if logging:
        energyhandle = data.EnergyStore(state=test_state)
        energy_steps = 20
        energyfn = energyhandle.update
    else:
        energy_steps = 0
        energyfn = None
    

    test_vaf_method = method.VelocityAutoCorrelation(state = test_state)
    test_gr_method = method.RadialDistributionPeriodicNVE(state = test_state, rsteps = 200)

    per_printer = data.PercentagePrinter(dt,t,10)

    schedule = method.Schedule([
                                plotsteps,
                                test_xyz_steps,
                                energy_steps,
                                1
                                ],
                               [
                                plotfn,
                                writefn,
                                energyfn,
                                per_printer.tick
                                ])

    # Create an integrator for above state class.
    test_integrator = method.VelocityVerlet(state = test_state, schedule=schedule)
    # test_integrator = method.VelocityVerletBox(state = test_state, schedule=schedule)


    ###########################################################

    test_integrator.integrate(dt=dt, t=t)

    if test_domain.halos is not False:
        test_domain.halos.timer.time("Total time in halo exchange.")
    test_state.timer.time("Total time in forces update.")
    test_state.cpu_forces_timer.time("Total time cpu forces update.")
    if gpucuda.INIT_STATUS():
        test_state.gpu_forces_timer.time("Total time gpu forces update.")
    ###########################################################
    # test_state.swaptimer.time("state time to swap arrays")
    

    #If logging was enabled, plot data.
    if (logging):
        energyhandle.plot()

    if runtime.MPI_HANDLE.rank ==0:
        try:

            # a=input("PRESS ENTER TO CONTINUE.\n")
            pass
        except:
            pass
    #MPI_HANDLE.barrier()
