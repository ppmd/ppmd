.. contents::

Examples
========

High-level Argon Example
~~~~~~~~~~~~~~~~~~~~~~~~

Domain
......

A domain is the physical space in which a simulation takes place. An instance of a domain class contains the extent of the cuboid enclosing all the particles. We initalise an empty periodic domain to contain our Argon atoms: ::
    
    periodic_domain = domain.BaseDomainHalo()



Position, Velocity and Mass
...........................

Init, mass pos, vel: ::

    position_init = simulation.PosInitLatticeNRhoRand(N, rho=0.2, dev=0.)
    velocity_init = simulation.VelInitNormDist(mu=0., sig=5.)
    mass_init = simulation.MassInitIdentical(m=39.948)



Potential
.........

An interaction between particles is an example of a pairwise operation. The following creates a Lennard-Jones potential with the parameters to describe an Argon system. ::
    
    ar_potential = potential.LennardJones(sigma=3.405, epsilon=0.9661, rc=8.5)


Combining into a simulation
...........................

Combining the above into a simulation. ::

    sim = simulation.BaseMDSimulation(domain_in=periodic_nve_domain,
                                      potential_in=ar_potential,
                                      particle_pos_init=position_init,
                                      particle_vel_init=velocity_init,
                                      particle_mass_init=mass_init,
                                      n=N)


xyz Writer
..........

Init method to write xyz trajectories using a specified state. ::

    xyz_writer = method.WriteTrajectoryXYZ(state=sim.state, 
                                           dir_name='./', 
                                           file_name='out.xyz')



schedule
........

How to schedule events within an integration. ::

    schedule = method.Schedule([5], [xyz_writer.write])


Create and run an integrator
............................

Create ::

    integrator = method.VelocityVerlet(simulation = sim, schedule=schedule)

run::

    integrator.integrate(dt=0.0001, t=0.1)

Overall code
............

All the above blocks of code together ::
    
    from ppmd import *

    # choose a domain.
    periodic_domain = domain.BaseDomainHalo()

    # set potential between particles.
    ar_potential = potential.LennardJones(sigma=3.405, epsilon=0.9661, rc=8.5)

    # N: Number of particles.
    n = 10
    N = n ** 3

    # position initialisation method
    position_init = simulation.PosInitLatticeNRhoRand(N, rho=0.2, dev=0.)

    # velocity initialisation method
    velocity_init = simulation.VelInitNormDist(mu=0., sig=5.)

    # mass initialisation method
    mass_init = simulation.MassInitIdentical(m=39.948)

    # Combine the existing intialisations into a simulation.
    sim = simulation.BaseMDSimulation(domain_in=periodic_domain,
                                      potential_in=ar_potential,
                                      particle_pos_init=position_init,
                                      particle_vel_init=velocity_init,
                                      particle_mass_init=mass_init,
                                      n=N
                                      )

    # Create a xyz writer.
    xyz_writer = method.WriteTrajectoryXYZ(state=sim.state, 
                                           dir_name='./', 
                                           file_name='out.xyz')

    # Write xyz trajectory every 5 timesteps.
    schedule = method.Schedule([5], [xyz_writer.write])

    # create an integrator instance.
    integrator = method.VelocityVerlet(simulation = sim, schedule=schedule)

    # integrate forward in time.
    integrator.integrate(dt=0.0001, t=0.1)



Low-level Argon Example
~~~~~~~~~~~~~~~~~~~~~~~~

Defining a custom pairwise operation
....................................

In the high level example the required potential is predefined in the package. The code below demonstrates how to recreate the Lennard-Jones potential using a kernel and pairloop combination.

Defining the Lennard-Jones interaction used in the high-level example as a custom pairwise operation. ::

    code here







