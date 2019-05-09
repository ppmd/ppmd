.. highlight:: python


Velocity-Verlet Integrator
==========================

In this example we will demonstrate how to loop over particle data in a manner that is typical of an time integrator. We shall assume that our system has the following ParticleDat data structures defined.

::
    
    # Note: positions stored in PositionDat
    A.P = PositionDat(ncomp=3, dtype=ctypes.c_double)
    A.V = ParticleDat(ncomp=3, dtype=ctypes.c_double)
    A.F = ParticleDat(ncomp=3, dtype=ctypes.c_double)
    A.M = ParticleDat(ncomp=1, dtype=ctypes.c_double)

As Velocity-Verlet is a two stage integrator we shall use two ParticleLoops, one for each stage of the algorithm. The first stage of Velocity-Verlet is a half timestep update of the velocities and a full timestep update of positions. We use superscripts to denote timesteps and subscripts to denote particle indices:

.. math::

  V^{n+\frac{1}{2}}_{i} = V^{n}_{i} + \frac{\delta t}{2} \frac{F^{n}_{i}}{M_{i}}

  P^{n+1}_{i} = P^{n}_{i} + \delta t V^{n+\frac{1}{2}}_{i}

Between stage one described above and the second stage a PairLoop will typically be used to update the forces using the positions we updated in the first stage. Assuming up-to-date forces the second stage of Velocity-Verlet updates the velocites using the newly computed forces.

.. math::

  V^{n+1}_{i} = V^{n+\frac{1}{2}}_{i} + \frac{\delta t}{2} \frac{F^{n + 1}_{i}}{M_{i}}


We write a kernel for each stage as a portion of C code in a Python string:

::

    # -- Stage 1 --
    vv_kernel1_code = '''
    const double M_tmp = 1.0 / M.i[0];
    V.i[0] += dht * F.i[0] * M_tmp;
    V.i[1] += dht * F.i[1] * M_tmp;
    V.i[2] += dht * F.i[2] * M_tmp;
    P.i[0] += dt * V.i[0];
    P.i[1] += dt * V.i[1];
    P.i[2] += dt * V.i[2];
    '''

    # -- Stage 2 --
    vv_kernel2_code = '''
    const double M_tmp = 1.0 / M.i[0];
    V.i[0] += dht * F.i[0] * M_tmp;
    V.i[1] += dht * F.i[1] * M_tmp;
    V.i[2] += dht * F.i[2] * M_tmp;
    '''

In the two C kernels above we use values (:code:`dt` = :math:`\delta t` and :code:`dht` = :math:`\frac{\delta t}{2}`) which, in our example, are constant. We declare constant values using :class:`kernel.Constant` instances as follows:

::

    dt = 0.0001
    constants = [
        kernel.Constant('dt', dt),
        kernel.Constant('dht', 0.5*dt),
    ]

We may then combine the strings containing the kernel code with the declared constants to create a :class:`kernel.Kernel` instance that describes the operations that we wish to perform in a container that can be passed to a looping method. For a first argument we pass a user specified name to aid profiling, names are user chosen and are not required to contain any information:

::

    vv_kernel1 = kernel.Kernel('vv1', vv_kernel1_code, constants)
    vv_kernel2 = kernel.Kernel('vv2', vv_kernel2_code, constants)




The final step is to create two :class:`ParticleLoop` instances, one for each kernel. Each ParticleLoop is constructed with a kernel and a Python dictionary refered to as the :code:`dat_dict`. The dictionary matches the symbols used in the C kernel with the corresponding data structure, the data structures are called as show below with an access descriptor. By using access descriptors we indicate to the ParticleLoop how the kernel will access data.
::

    loop1 = ParticleLoop(
        kernel=vv_kernel1,
        dat_dict={'P': A.P(md.access.W),
                  'V': A.V(md.access.W),
                  'F': A.F(md.access.R),
                  'M': A.M(md.access.R)}
    )

    loop2 = ParticleLoop(
        kernel=vv_kernel2,
        dat_dict={'V': A.V(md.access.W),
                  'F': A.F(md.access.R),
                  'M': A.M(md.access.R),
                  'k': A.KE(md.access.INC0)}
    )



To execute our two kernels over all particles we call the :code:`execute` method on each ParticleLoop instance:

::

    # Execute kernel 1
    loop1.execute()

    # -- update forces using a pairloop --

    # Execute kernel 2
    loop2.execute()


