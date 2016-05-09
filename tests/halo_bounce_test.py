#!/usr/bin/python

import numpy as np
from ppmd import *
import getopt, sys

directions = [
              [1, 0, 0],
              [1, 1, 0],
              [1, -1, 0],
              [1, 0, 1],
              [1, 0, -1],
              [0, 1, 1],
              [0, 1, -1],
              [1, 1, 1],
              [-1, 1, 1],
              [1, -1, 1],
              [-1, -1, 1]
             ]






direc = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], "N:")
except getopt.GetoptError as err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    sys.exit(2)
for o, a in opts:
    if o == "-N":
        direc=int(a)
    else:
        assert False, "unhandled option"


# import runtime
# debug level
runtime.DEBUG.level = 3
# verbosity level
runtime.OPT.level = 3

runtime.VERBOSE.level = 3
# timer level
runtime.TIMER.level = 3
# build timer level
runtime.BUILD_TIMER.level = 3

# cuda on/off
runtime.CUDA_ENABLED.flag = False

for direc in directions:
    cell.reset_cell_list()
    halo.reset_halos()

    print "DIRECTION", direc

    # plot as computing + at end?
    plotting = True

    N = 2

    t = 0.03
    dt = 0.00001

    # See above
    test_domain = domain.BaseDomainHalo()
    test_potential = potential.LennardJones(sigma=1.0, epsilon=1.0, rc=2.5)


    # Set alternating masses for particles.
    test_mass_init = simulation.MassInitTwoAlternating(1., 1.)

    # Give first two particles specific velocities
    test_vel_init = simulation.VelInitTwoParticlesInABox(vx=np.array([0., 0., 0.]), vy=np.array([0., 0., 0.]))


    # Initialise two particles on an axis a set distance apart.
    test_pos_init = simulation.PosInitTwoParticlesInABox(rx=0.35, extent=np.array([30., 30., 30.]), axis=np.array(direc))

    # Create simulation class from above initialisations.

    sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                                       potential_in=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N,
                                       domain_boundary_condition=domain.BoundaryTypePeriodic()
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

    per_printer = method.PercentagePrinter(dt,t,20)

    schedule = method.Schedule([
                                plotsteps,
                                1
                                ],
                               [
                                plotfn,
                                per_printer.tick
                                ])

    # Create an integrator for above state class.
    test_integrator = method.VelocityVerlet(simulation = sim1, schedule=schedule)

    test_integrator.integrate(dt=dt, t=t)

    plothandle.cleanup()


