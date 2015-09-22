#!/usr/bin/python


from ppmd import *

# choose a domain.
periodic_nve_domain = domain.BaseDomainHalo()

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
sim = simulation.BaseMDSimulation(domain_in=periodic_nve_domain,
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

# Write xyz trajectory every 50 timesteps.
schedule = method.Schedule([5], [xyz_writer.write])



# create an integrator instance.
integrator = method.VelocityVerlet(simulation = sim, schedule=schedule)

# integrate forward in time.
integrator.integrate(dt=0.0001, t=0.1)






