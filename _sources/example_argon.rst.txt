.. highlight:: python


Low-level Argon Example
~~~~~~~~~~~~~~~~~~~~~~~~

We start by importing required modules and the package under the name ``md``. We create aliases to certain classes that may need to be altered to run on CUDA capable GPUs.
::

    #!/usr/bin/python
    
    import ctypes
    import numpy as np
    import math
    import os

    import ppmd as md
    

    rank = md.mpi.MPI_HANDLE.rank

    PositionDat = md.data.PositionDat
    ParticleDat = md.data.ParticleDat
    ScalarArray = md.data.ScalarArray
    State = md.state.State
    ParticleLoop = md.loop.ParticleLoop
    Pairloop = md.pairloop.PairLoopNeighbourListNS



In this example we will create random positions and velocities on rank 0 then distribute these across the available processes.
::

    N = 1000

    # create a state called A
    A = State()
    
    # set the number of particles in the system to be N
    A.npart = N

    # Initialise a domain with extents 8x8x8 with periodic boundaries.
    A.domain = md.domain.BaseDomainHalo(extent=np.array([8., 8., 8.]))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    
    # In our system each particle has a position, velocity, mass and acceleration, 
    # we indicate these properties by adding appropriate ParticleDats to the state
    # A
    
    # We indicate to the framework which property contains our positions with a
    # specific data type called PositionDat.
    
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.mass = ParticleDat(ncomp=1)

    # As ParticleDats default to containing double values, when we want a dat that
    # stores a golbal id we add an additional parameter to indicate that this dat
    # contains int type values.
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    # We add global properties to our state through ScalarArray objects. Values in
    # these structures are not tied to any particular particle. Here we add a 
    # ScalarArray for the potential energy `u`.
    
    A.u = ScalarArray(ncomp=1, name='u')

    # Initialise the data structures with random values.
    # we will later use the values which were assigned on rank 0
    
    # sequential global ids.
    A.gid[:, 0] = np.arange(A.npart)
    A.p[:] = np.random.uniform(-4.0, 4.0, [N,3])
    A.v[:] = np.random.normal(size=[10,3])
    A.mass[:] = 1.0

    A.u[0] = 0.0


    # distribute the data on rank 0 to the reset of the system.
    A.scatter_data_from(0)



For this example we recreate the Buckingham potential and set constants suitable for modelling Argon.
::
    
    kernel_code = '''
    const double R0 = P.j[0] - P.i[0];
    const double R1 = P.j[1] - P.i[1];
    const double R2 = P.j[2] - P.i[2];
    
    const double r2 = R0*R0 + R1*R1 + R2*R2;
    
    const double r = sqrt(r2);
    // \\exp{-B*r}
    const double exp_mbr = exp(_MB*r);
    
    // r^{-2, -4, -6}
    const double r_m1 = 1.0/r;
    const double r_m2 = r_m1*r_m1;
    const double r_m4 = r_m2*r_m2;
    const double r_m6 = r_m4*r_m2;
    
    // \\frac{C}{r^6}
    const double crm6 = _C*r_m6;
    
    // A \\exp{-Br} - \\frac{C}{r^6}
    u[0]+= (r2 < rc2) ? 0.5*(_A*exp_mbr - crm6 + internalshift) : 0.0;
    
    // = AB \\exp{-Br} - \\frac{C}{r^6}*\\frac{6}{r}
    const double term2 = crm6*(-6.0)*r_m1;
    const double f_tmp = _AB * exp_mbr + term2;
    
    F.i[0]+= (r2 < rc2) ? f_tmp*R0 : 0.0;
    F.i[1]+= (r2 < rc2) ? f_tmp*R1 : 0.0;
    F.i[2]+= (r2 < rc2) ? f_tmp*R2 : 0.0;
    '''
    
    a=1.69*10**-8.0,
    b=1.0/0.273,
    c=102*10**-12,

    rc = 2.5
    rn = 1.1 * rc

    shift_internal = -1.0 * a * math.exp(b*(-1.0/rc)) + c*(-1.0/(rc**6.0))
    
    kernel_constants = (
        kernel.Constant('_A', a),
        kernel.Constant('_AB', a*b),
        kernel.Constant('_B', b),
        kernel.Constant('_MB', -1.0*b),
        kernel.Constant('_C', c),
        kernel.Constant('rc2', rc ** 2),
        kernel.Constant('internalshift', shift_internal)
    )
    
    B_kernel = md.kernel.Kernel('Buckingham', kernel_code, kernel_constants, ['math.h'])
    
    
    # pairloop to update forces and potential energy
    force_updater = Pairloop(
        kernel=B_kernel,
        dat_dict={
            'P': A.p(md.access.R),
            'F': A.f(md.access.INC0),
            'u': A.u(md.access.INC0)
        },
        shell_cutoff=rn
    )

To integrate the system forward in time we will define a Velocity Verlet integrator using two `ParticleLoop` instances.
::

    dt = 0.0001

    vv_kernel1_code = '''
    const double M_tmp = 1.0 / M.i[0];
    V.i[0] += dht * F.i[0] * M_tmp;
    V.i[1] += dht * F.i[1] * M_tmp;
    V.i[2] += dht * F.i[2] * M_tmp;
    P.i[0] += dt * V.i[0];
    P.i[1] += dt * V.i[1];
    P.i[2] += dt * V.i[2];
    '''

    vv_kernel2_code = '''
    const double M_tmp = 1.0 / M.i[0];
    V.i[0] += dht * F.i[0] * M_tmp;
    V.i[1] += dht * F.i[1] * M_tmp;
    V.i[2] += dht * F.i[2] * M_tmp;
    '''
    constants = [
        md.kernel.Constant('dt', dt),
        md.kernel.Constant('dht', 0.5*dt),
    ]

    vv_kernel1 = md.kernel.Kernel('vv1', vv_kernel1_code, constants)
    vv_p1 = ParticleLoop(
        kernel=vv_kernel1,
        dat_dict={'P': A.p(md.access.W),
                  'V': A.v(md.access.W),
                  'F': A.f(md.access.R),
                  'M': A.mass(md.access.R)}
    )

    vv_kernel2 = md.kernel.Kernel('vv2', vv_kernel2_code, constants)
    vv_p2 = ParticleLoop(
        kernel=vv_kernel2,
        dat_dict={'V': A.v(md.access.W),
                  'F': A.f(md.access.R),
                  'M': A.mass(md.access.R)}
    )


By using a custom Python iterator `IntegratorRange` the framework internally controls when intermediate lists are rebuilt. By passing our velocity dat and a timestep the framework is able to monitor maximum velocities to ensure particles do not move too far between list updates. The penultimate parameter defines the maximum number of iterations to keep internal lists for and the last parameter declares the distance between the cutoff used by the pairloop and the cutoff used by the potential.
::

    u_list = []    
    
    for it in md.method.IntegratorRange(1000, dt, A.v, 10, 0.25):

        vv_p1.execute()
        force_updater.execute()
        vv_p2.execute()

        if it % 10 == 0:
            u_list.append(A.u[0])
    
    u_array = md.mpi.all_reduce(np.array(u_list))

Boundary conditions are automatically applied by the framework using the domain and boundary condition declared earlier.











