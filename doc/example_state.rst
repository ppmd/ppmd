.. highlight:: python



State Containers
================

The state object couples multiple ParticleDats with a simulation domain that notionally contains the particles. The framework uses a spatial domain decomposition approach for course grain parallelism across MPI ranks. Spatial decomposition requires that when a particle leaves a subdomain owned by a process the data associated with the particle is moved to the new process.

By adding ParticleDat instances to a state object boundary conditions and the movement of data between MPI ranks can be handled automatically by the framework. Instances of state objects follow the convention that ``npart`` returns the total number of particles in the system. ``npart_local`` returns the number of particles currently stored on the calling MPI rank.


Base Case
~~~~~~~~~

The base case state object is designed with NVE ensembles in mind. In the example below we create a state object called ``A`` and set the total number of particles in the system.

::

    from ppmd import *
    State = state.State
    PBC = domain.BoundaryTypePeriodic

    A = State()
    A.npart = 1000


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



Initialise Data Scatter Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



