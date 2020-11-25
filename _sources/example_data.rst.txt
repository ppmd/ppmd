.. highlight:: python


Particle Data
=============

We provide a flexible data structure to store data on a per-particle basis. 
The number of elements per particle and the data type are specified by the user.

Particle positions are stored in a particular type of ParticleDat called a PositionDat.

Standalone ParticleDat and PositionDat instances are not particularly useful as they do not automatically group the data for particles together.
To group particle data please refer to the :ref:`State Containers` section.



ParticleDat Object
~~~~~~~~~~~~~~~~~~


Particle properties are stored within :class:`~data.ParticleDat` containers. These can be considered as two dimensional matrices with each row storing data for a particle. For example the velocities of :math:`N` particles would be stored within a :math:`N` X :math:`3` ParticleDat.

::

    from ppmd import *
    ParticleDat = data.ParticleDat
    vel = ParticleDat(npart=N, ncomp=3)


ParticleDats may contain ctypes.c_double, ctypes.c_int and ctypes.c_long data. To specifiy a data type other than the default ctypes.c_double the ``dtype`` keyword should be used.

::

    import ctypes
    num_neighbours = ParticleDat(npart=N, ncomp=1, dtype=ctypes.c_int)

These data structures are specialised wrappers around Numpy objects and the underlying numpy data may be accessed by using the standard Python\Numpy indexing. For example to initialise the ``vel`` ParticleDat above with values drawn from a standard normal distribution we may do the following.

::

    import numpy as np
    vel[:,:] = np.random.normal(size=(N,3))

To create a ``CUDA`` ParticleDat use the ParticleDat from ``ppmd.cuda``:

::

    from ppmd.cuda import *
    ParticleDat = cuda_data.ParticleDat
    vel = ParticleDat(npart=N, ncomp=3)
    vel[:,:] = np.random.normal(size=(N,3))

The code above creates a ParticleDat on a CUDA device and assigns the values drawn from a standard normal distribution. The copying of data between the host and the CUDA device is handled automatically by the framework.


PositionDat Object
~~~~~~~~~~~~~~~~~~

PositionDats always have 3 components and are of type `ctypes.c_double` and can be initialised with

::
    
    pos = PositionDat()













