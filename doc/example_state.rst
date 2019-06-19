.. highlight:: python



State Containers
================

The state object couples multiple ParticleDats with a simulation domain that notionally contains the particles. The framework uses a spatial domain decomposition approach for course grain parallelism across MPI ranks. Spatial decomposition requires that when a particle leaves a subdomain owned by a process the data associated with the particle is moved to the new process.

By adding ParticleDat instances to a state object boundary conditions and the movement of data between MPI ranks can be handled automatically by the framework. Instances of state objects follow the convention that ``npart`` returns the total number of particles in the system. ``npart_local`` returns the number of particles currently stored on the calling MPI rank.


Base Case
~~~~~~~~~

The base case state object is designed with NVE ensembles in mind. In the example below we create a state object called ``A``.

::

    from ppmd import *
    State = state.State
    PBC = domain.BoundaryTypePeriodic

    A = State()


Here we add boundary conditions to the state called ``A``.

::

    A.domain = domain.BaseDomainHalo(extent=(10., 10., 10.))
    A.domain.boundary_condition = PBC()

The second requirement for a state object is that a particular type of ParticleDat is added called a PositionDat. With a domain and a PositionDat the state object can handle the movement of particle data between processes.

::

    PositionDat = data.PositionDat
    A.pos = PositionDat(ncomp=3)

When added ParticleDats to a state class the ``npart`` argument should not be passed. As the framework uses spatial domain decomposition the ParticleDat holds valid particle data in the first ``A.npart_local`` rows.

::

    print A.pos[:A.npart_local:, ::]


Particle Data Operations
~~~~~~~~~~~~~~~~~~~~~~~~

When we add ``ParticleDats`` to a state we define which properties particles hold. For example the following code defines properties for positions, velocities, forces and global ids.

::

    A.pos = PositionDat(ncomp=3)
    A.vel = ParticleDat(ncomp=3)
    A.force = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int64)


Particles are added through the ``State.modify()`` interface. This interface allows particles to be added and removed. The use of this interface is collective over all MPI ranks. A "window" where particles can be added and removed is created, on all MPI ranks, with:

::
    
    with A.modify() as m: # this line must be called on all MPI ranks.
        # Add/remove here


Adding Particles
----------------

After ``State.modify()`` is called particle data is added with a call to ``add``. In the following example with add ``N=1000`` particles with uniform random positions, normal distributed forces and global ids. Particle properties where no values are passed are initialised to zero.

::

    N = 1000
    with A.modify() as m:
        if A.domain.comm.rank == 0:
            m.add({
                A.pos: np.random.uniform(-5, 5, size=(N, 3)),
                A.vel: np.random.normal(0, 1, size=(N, 3)),
                A.gid: np.arange(N).reshape((N, 1))
            })
    
    print(A.npart) # should print 1000
    print(A.npart_local) # should print the number of particles on this MPI rank



The ``if A.domain.comm.rank == 0`` statement ensures that only rank 0 adds particles. There are no restrictions on which MPI ranks may add particles. Furthermore a MPI rank can add particles anywhere in the simulation domain. Particle data is added on completion of the ``State.modify()`` context.
Particles are automatically communicated to the MPI rank that owns the subdomain for those particular particles.


Removing Particles
------------------

Particles are removed by passing local indices to the ``remove`` call of (in this case) ``m``. For example, to remove the first two particles on rank 1:

::

    with A.modify() as m:
        if A.domain.comm.rank == 1:
            m.remove((0, 1))

It is the users responsibility to determine the local index of the particles they wish to remove. Particles are removed on completion of the ``State.modify()`` context. Removing particles will force a reordering of particles.


Modifying Particle Data
-----------------------

The data held in a ``ParticleDat`` is modified in a similar manner to modifying a ``State``. First a collective call is made to ``ParticleDat.modify_view()`` on all MPI ranks. This call returns a ``NumPy`` view of size ``A.npart_local x ParticleDat.ncomp`` which can be modified safely. For example to draw new random velocities we perform:

::

    with A.vel.modify_view() as m:
        m[:, :] = np.random.normal(0, 1, size=(A.npart_local, 3))


If particle positions are modified then the particles will be communicated between MPI ranks on completion of the ``ParticleDat.modify_view`` context. This may well cause particles to change MPI ranks and hence cause a reordering of particle data.



Initialise Data Scatter Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scattering data is a legacy method to broadcast particle data across the MPI ranks. Using ``State.modify()`` is the replacement.

In this example we will create initial data on rank 0 then scatter that data across available MPI ranks. When scattering data from a rank the total number of particles ``State.npart`` should be set on the state object prior to scattering.

::

    import numpy as np
    import ctypes
    from ppmd import *

    # aliases start

    State = state.State
    ParticleDat = data.ParticleDat
    PositionDat = data.PositionDat
    PBC = domain.BoundaryTypePeriodic

    # aliases end


    N = 1000
    A = State()

    # Total number of particles is set.
    A.npart = N

    A.domain = domain.BaseDomainHalo(extent=(10., 10., 10.))
    A.domain.boundary_condition = PBC()

    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)


    if mpi.MPI_HANDLE.rank == 0:
        A.p[:] = utility.lattice.cubic_lattice((10, 10, 10), (10., 10., 10.))
        A.v[:] = np.random.normal(0.0, 1.0, size=(N, 3))
        A.f[:] = 0.0
        A.gid[:, 0] = np.arange(N)

    A.scatter_data_from(0)


The state will use the positions in the PositionDat to filter which data belongs on which subdomain.



