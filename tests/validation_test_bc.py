#!/usr/bin/python

from ppmd import *

#import runtime
# debug level
runtime.DEBUG.level = 3
#verbosity level
runtime.OPT.level = 3


runtime.VERBOSE.level = 3
#timer level
runtime.TIMER.level = 3
#build timer level
runtime.BUILD_TIMER.level = 3




#cuda on/off
runtime.CUDA_ENABLED.flag = False




import numpy as np



# plot as computing + at end?
plotting = True
    

t=1.
dt=0.0001




N = 1

# See above
test_domain = domain.BaseDomainHalo()
test_potential = potential.LennardJones(1.,1.,0.01)


test_pos_init = simulation.PosInitOneParticleInABox(r = np.array([0.0, 0., 0.]), extent = np.array([0.3, 0.3, 0.3]))

test_vel_init = simulation.VelInitOneParticleInABox(vx = np.array([0., -5., 5.]))

test_mass_init = simulation.MassInitIdentical(5.)


sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                               potential_in=test_potential,
                               particle_pos_init=test_pos_init,
                               particle_vel_init=test_vel_init,
                               particle_mass_init=test_mass_init,
                               n=N
                               )

if mpi.MPI_HANDLE.rank == 1:
    _sfd = test_domain.get_shift()
    print "SHIFT START", mpi.MPI_HANDLE.rank
    for ix in range(26):
        print ix, _sfd[ix*3:(ix+1)*3:]

    print "SHIFT END", mpi.MPI_HANDLE.rank



# plotting handle
plothandle = method.DrawParticles(state=sim1.state)
plotsteps = 10
plotfn = plothandle.draw


per_printer = method.PercentagePrinter(dt,t,10)

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


###########################################################

test_integrator.integrate(dt=dt, t=t)

sim1.state.move_timer.stop("move total time")
sim1.state.compress_timer.stop("compress time")


sim1.timer.time("Total time in forces update.")
sim1.cpu_forces_timer.time("Total time cpu forces update.")
###########################################################
# sim1.state.swaptimer.time("state time to swap arrays")

pio.pprint("LoopTimer resolution: ", opt.get_timer_accuracy(), "s")
pio.pprint("Recorded forces cputime from looping ", sim1._forces_update_lib.loop_timer.cpu_time)

