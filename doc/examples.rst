.. contents::
.. highlight:: python
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

Setup
.....

In the high level example the required potential is predefined in the package. The code below demonstrates how to recreate the Lennard-Jones potential using a kernel and pairloop combination.

The initial setup procedure is similar to the high-level example. We create an instance of the simulation class with a domain and methods to initialise the individual particles. ::

    from ppmd import *

    # choose a domain.
    periodic_domain = domain.BaseDomainHalo()

    # N: Number of particles.
    n = 10
    N = n ** 3

    # position initialisation method
    position_init = simulation.PosInitLatticeNRhoRand(N, rho=0.2, dev=0.)

    # velocity initialisation method
    velocity_init = simulation.VelInitNormDist(mu=0., sig=5.)

    # mass initialisation method
    mass_init = simulation.MassInitIdentical(m=39.948)

    # Combine the existing intialisations into a simulation. We do not pass a potential, 
    # the cell structure is setup with a passed cutoff.

    sim = simulation.BaseMDSimulation(domain_in=periodic_domain,
                                      particle_pos_init=position_init,
                                      particle_vel_init=velocity_init,
                                      particle_mass_init=mass_init,
                                      n=N,
                                      cutoff=8.5
                                      )

Pairloop example: Custom potential
..................................

Here we define the Lennard-Jones interaction used in the high-level example as a custom pairwise operation. In principle a kernel is a block of code that is combined with a looping method to produce code that either loops over particle pairs or individual particles.

A kernel consists of a block of code describing the interaction and a map between the variables used in the kernel and the particle dats to loop over. Named constants can be replaced with their numerical values as an optimisation.

The kernel code is constructed as a string. For a pairwise interaction such as a potential the particle dats are presented to the kernel as pointer arrays with two elements. Such that for particle pair (i,j) and a particle dat labeled "P", the kernel would expect P[0] to point to the data for particle i and P[1] to point to the data for particle j.

.. code-block:: c

    kernel_code = '''
    const double R0 = P[1][0] - P[0][0]; // Distance in x direction between particles.
    const double R1 = P[1][1] - P[0][1]; // Distance in y direction between particles.
    const double R2 = P[1][2] - P[0][2]; // Distance in z direction between particles.

    const double r2 = R0*R0 + R1*R1 + R2*R2;

    if (r2 < rc2){

        const double r_m2 = sigma2/r2;
        const double r_m4 = r_m2*r_m2;
        const double r_m6 = r_m4*r_m2;
        
        u[0]+= CV*((r_m6-1.0)*r_m6 + 0.25);
        
        const double r_m8 = r_m4*r_m4;
        const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

        
        A[0][0]+=f_tmp*R0;
        A[0][1]+=f_tmp*R1;
        A[0][2]+=f_tmp*R2;
        
        A[1][0]-=f_tmp*R0;
        A[1][1]-=f_tmp*R1;
        A[1][2]-=f_tmp*R2;

    }
    '''

Constants can be hardcoded into generated code by declaring values when constructing the kernel. All instances of the constants in the kernel code are replaced by the numerical values of the constants.

.. code-block:: python

    sigma = 3.405
    epsilon = 0.9661
    cutoff = 8.5

    kernel_constants = (kernel.Constant('sigma2', sigma ** 2),
                        kernel.Constant('rc2', cutoff ** 2),
                        kernel.Constant('CF', -48 * epsilon / sigma ** 2),
                        kernel.Constant('CV', 4. * epsilon))

The kernel code is combined with the kernel constants to create a :class:`~kernel.Kernel` instance. User defined header files along with non pointer arguments may also be included in the creation of a kernel.

.. code-block:: python

    LJ_kernel = kernel.Kernel('custom_lennard_jones', 
                              kernel_code, 
                              kernel_constants)

The final part of the kernel is the map between the variables used in the kernel code and the particle dats in the simulation state. These are defined as a python dictonary and are passed with the kernel to a looping method. The access descriptors declare to the looping method the access type required by the kernel to the data.

.. code-block:: python

    kernel_dat_dict = {'P': sim.state.positions(access.R), # Read only access
                       'A': sim.state.forces(access.W),    # Write only access
                       'u': sim.state.u(access.INC)}       # Incremental access

After passing the kernel to a looping method the C code is generated based on the user kernel. Here a cell based looping method is chosen for this potential interaction.

.. code-block:: python

    force_update_pairloop = pairloop.PairLoopRapaportHalo(domain=periodic_domain,
                                                          kernel=LJ_kernel,
                                                          dat_dict=kernel_dat_dict)

The pair loop can be executed by calling: ::
    
    force_update_pairloop.execute()


Paricle loop example: Velocity-Verlet integrator
................................................

The kernel is created in the same way as a pair loop kernel. The kernel and particle dat dictonary is passed to a looping method that loops over all particles once. A pointer is created to the position of the current particle in each of the particle dats. 

.. code-block:: python

    vv_kernel1_code = '''
    const double M_tmp = 1/M[0];
    V[0] += dht*A[0]*M_tmp;
    V[1] += dht*A[1]*M_tmp;
    V[2] += dht*A[2]*M_tmp;
    P[0] += dt*V[0];
    P[1] += dt*V[1];
    P[2] += dt*V[2];
    '''
            
    vv_kernel2_code = '''
    const double M_tmp = 1/M[0];
    V[0] += dht*A[0]*M_tmp;
    V[1] += dht*A[1]*M_tmp;
    V[2] += dht*A[2]*M_tmp;
    '''

    dt = 0.0001
    vv_constants = (kernel.Constant('dt', dt), 
                    kernel.Constant('dht',0.5 * dt))
                    
    vv_dat_dict = {'P':sim.state.positions(access.RW),
                   'V':sim.state.velocities(access.RW),
                   'A':sim.state.forces(access.R),
                   'M':sim.state.mass(access.R)}

    vv_kernel1 = kernel.Kernel('vv1', vv_kernel1_code, vv_constants)
    vv_kernel2 = kernel.Kernel('vv2', vv_kernel2_code, vv_constants)

    vv_part1_loop = loop.SingleAllParticleLoop(sim.state.as_func('n'), 
                                               sim.state.types,
                                               vv_kernel1,
                                               vv_dat_dict)

    vv_part2_loop = loop.SingleAllParticleLoop(sim.state.as_func('n'),
                                               sim.state.types,
                                               vv_kernel2,
                                               vv_dat_dict)



Resulting simulation
....................

Using the pair loop to update the forces on each particle and the two particle loops to implement a time stepping method the system can be integrated forward in time.

.. code-block:: python

    for ix in range(100):

        # first part of velocity verlet.
        vv_part1_loop.execute()
        
        # To be handled by access descriptiors in future.
        periodic_domain.bc_execute()
        
        
        # update forces using custom potential.
        
        # To be handled by access descriptiors in future.
        cell.cell_list.sort()
        sim.state.positions.halo_exchange()
        
        
        # update forces and potential energy using custom pairloop.
        force_update_pairloop.execute()

        
        # second part of velocity verlet.
        vv_part2_loop.execute()
        
    
    
    
    
    
    
    
    
    


























