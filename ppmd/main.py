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


import mpi
import domain
import potential
import state
import numpy as np
import method
import data
import gpucuda

import os
import simulation
import halo



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
    logging = True

    # Write XYZ?
    writing = False


    t=0.0001*24
    #t=0.0024
    #t=0.0174
    t=0.1
    dt=0.0001


    if test_1000:
        # n=25 reasonable size
        n = 50
        N = n**3
        # n=860
        rho = 1.
        mu = 1.0
        nsig = 5.0
        
        # Initialise basic domain
        test_domain = domain.BaseDomainHalo()

        # Initialise LJ potential
        test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)    
        
        # Place n particles in a lattice with given density.
        # test_pos_init = state.PosInitLatticeNRho(n, rho, None)
        test_pos_init = simulation.PosInitLatticeNRhoRand(N,rho,0.,None)
        
        # Normally distributed velocities.
        test_vel_init = simulation.VelInitNormDist(mu,nsig)
        # test_vel_init = state.VelInitPosBased()
        
        # Initialise masses, in this case sets all to 1.0.
        test_mass_init = simulation.MassInitIdentical(1.)
        # test_mass_init = state.MassInitTwoAlternating(100., 100.)
        
        
        
    if test_2_bounce:
        N = 2
        dt=0.00001
        
        # See above
        test_domain = domain.BaseDomainHalo()
        test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)


        # Initialise two particles on an axis a set distance apart.
        test_pos_init = simulation.PosInitTwoParticlesInABox(rx = 0.4, extent = np.array([12., 12., 12.]), axis = np.array([0,1,0]))

        # Give first two particles specific velocities
        test_vel_init = simulation.VelInitTwoParticlesInABox(vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.]))

        # Set alternating masses for particles.
        
        test_mass_init = simulation.MassInitTwoAlternating(1., 2.)
        
    if t_1_particle:
        
        N = 1
        
        # See above
        test_domain = domain.BaseDomainHalo()
        test_potential = potential.NULL(rc = 0.05)
        
        print test_potential.rc

        # Initialise two particles on an axis a set distance apart.
        test_pos_init = simulation.PosInitOneParticleInABox(r = np.array([0.0290, 0., 0.]), extent = np.array([0.3, 0.3, 0.3]))
        
        # Give first two particles specific velocities
        test_vel_init = simulation.VelInitOneParticleInABox(vx = np.array([5., 0., 0.]))
        
        # Set alternating masses for particles.
        
        test_mass_init = simulation.MassInitIdentical(5.)


    # Create simulation class from above initialisations.

    sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                                       potential_in=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N
                                       )

    # plotting handle
    if plotting:
        plothandle = method.DrawParticles(state=sim1.state)
        plotsteps = 100
        plotfn = plothandle.draw
    else:
        plothandle = None
        plotsteps = 0
        plotfn = None

    # xyz writing handle
    if writing:
        test_xyz = method.WriteTrajectoryXYZ(state=sim1.state)
        test_xyz_steps = 20
        writefn = test_xyz.write
    else:
        test_xyz_steps = 0
        writefn = None

    # xyz writing handle
    if logging:
        energyhandle = method.EnergyStore(state=sim1.state)
        energy_steps = 10
        energyfn = energyhandle.update
    else:
        energy_steps = 0
        energyfn = None
    

    test_vaf_method = method.VelocityAutoCorrelation(state = sim1.state)
    test_gr_method = method.RadialDistributionPeriodicNVE(state = sim1.state, rsteps = 200)

    per_printer = method.PercentagePrinter(dt,t,10)

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
    test_integrator = method.VelocityVerlet(simulation = sim1, schedule=schedule)
    # test_integrator = method.VelocityVerletBox(state = sim1.state, schedule=schedule)


    ###########################################################

    test_integrator.integrate(dt=dt, t=t)

    sim1.state.move_timer.stop("move total time")
    sim1.state.move_timer2.stop("move timer 2")
    sim1.state.compress_timer.stop("compress time")


    if test_domain.halos is not False:
        halo.HALOS.timer.time("Total time in halo exchange.")
    sim1.timer.time("Total time in forces update.")
    sim1.cpu_forces_timer.time("Total time cpu forces update.")
    if gpucuda.INIT_STATUS():
        sim1.gpu_forces_timer.time("Total time gpu forces update.")
    ###########################################################
    # sim1.state.swaptimer.time("state time to swap arrays")
    

    #If logging was enabled, plot data.
    if (logging):
        try:
            energyhandle.plot()
        except:
            pass

    if mpi.MPI_HANDLE.rank ==0:
        try:

            #a=input("PRESS ENTER TO CONTINUE.\n")
            pass
        except:
            pass
    #MPI_HANDLE.barrier()
