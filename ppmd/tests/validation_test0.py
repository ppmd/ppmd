#!/usr/bin/python

import runtime
# debug level
runtime.DEBUG.level = 0
#verbosity level
runtime.VERBOSE.level = 3
#timer level
runtime.TIMER.level = 3
#build timer level
runtime.BUILD_TIMER.level = 3


#cuda on/off
runtime.CUDA_ENABLED.flag = True

import os

from ppmd import domain
from ppmd import potential
from ppmd import method
from ppmd import simulation
from ppmd import io
import numpy as np

if __name__ == '__main__':

    file_dir = '../second_comparison/test_case_0/'


    # open the field file
    fh = open(file_dir + '/FIELD', 'r')
    _field = fh.read().split()
    fh.close()

    # get number of moecules from field file
    N = int(_field[_field.index('NUMMOLS') + 1])
    print 'N:', N

    epsilon = float(_field[_field.index('lj') + 1])
    sigma = float(_field[_field.index('lj') + 2])

    print "LJ: epsilon:", epsilon, "sigma:", sigma


    mass = float(_field[_field.index('Ar') + 1])


    # open the control file
    fh = open(file_dir + '/CONTROL', 'r')
    _control = fh.read().split()
    fh.close()

    dt = float(_control[_control.index('timestep') + 1])
    t = float(_control[_control.index('steps') + 1]) * dt
    rc = float(_control[_control.index('cutoff') + 1])

    print "rc =", rc

    # Initialise basic domain
    test_domain = domain.BaseDomainHalo()

    # Initialise LJ potential
    test_potential = potential.LennardJones(sigma=sigma,epsilon=epsilon, rc=rc)

    # Initialise masses
    test_mass_init = simulation.MassInitIdentical(mass)

    # Initialise positions and velocities
    test_pos_init = simulation.PosInitDLPOLYConfig(file_dir + '/CONFIG')
    test_vel_init = simulation.VelInitDLPOLYConfig(file_dir + '/CONFIG')

    # Create simulation class from above initialisations.
    sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                                       potential_in=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N
                                       )


    # Helper methods
    per_printer = method.PercentagePrinter(dt,t,10)
    pos_print = method.ParticleTracker(sim1.state.positions, 627, file_dir + '/pos.track')
    vel_print = method.ParticleTracker(sim1.state.velocities, 627, file_dir + '/vel.track')
    for_print = method.ParticleTracker(sim1.state.forces, 627, file_dir + '/for.track')

    tick = 5

    schedule = method.Schedule([1, tick, tick, tick], [per_printer.tick, pos_print.write, vel_print.write, for_print.write])

    # Create an integrator for above state class.
    test_integrator = method.VelocityVerlet(simulation=sim1, schedule=schedule)



    io.ParticleDat_to_xml(sim1.state.positions, file_dir + 'ppmd_x0.xml')

    # Check ParticleDat dump is correct
    test = io.xml_to_ParticleDat(file_dir + 'ppmd_x0.xml')
    for ix in range(N):
        assert np.all(test.dat[ix,0:3:] == sim1.state.positions.dat[ix,0:3:])



    ###########################################################

    test_integrator.integrate(dt=dt, t=t)

    sim1.timer.time("Total time in forces update.")
    sim1.cpu_forces_timer.time("Total time cpu forces update.")

    ###########################################################


    io.ParticleDat_to_xml(sim1.state.positions, file_dir + 'ppmd_x1.xml')

    # check ParticleDat dump is correct.
    test = io.xml_to_ParticleDat(file_dir + 'ppmd_x1.xml')
    for ix in range(N):
        assert np.all(test.dat[ix,0:3:] == sim1.state.positions.dat[ix,0:3:])








