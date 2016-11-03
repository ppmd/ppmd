.. highlight:: python


Kernels
=======

Kernels describe to a looping method what computation should be performed. A :code:`ParticleLoop` will execute a kernel on each particle in the dictonary of :code:`ParticleDat` instances that are passed to the constructor. In a :code:`ParticleLoop` a kernel may only read and write data to the particle currently under consideration or a :code:`ScalarArray`. A :code:`PairLoop` kernel follows the same rules as a :code:`ParticleLoop` kernel except that a :code:`PairLoop` kernel may also read data from another particle. Kernels must take into account that the order of execution over the set of particles or pairs of particles is not guaranteed.


Constants
~~~~~~~~~

We can substitute constant variables for "hardcoded" values by creating a new :class:`kernel.Constant`. For example to make :code:`dht` take the value :code:`0.0005` we would write:
::

    dht_c = kernel.Constant('dht', 0.0005)

:code:`ParticleLoop` and :code:`PairLoop` constructors expect to be passed a list of constants if such substitutions are required.


ParticleLoop Example
~~~~~~~~~~~~~~~~~~~~


This example is modified from the second step of a Velocity-Verlet integrator. The kernel presented below reads values from :code:`ParticleDat` data structures labeled :code:`F`, :code:`M` and updates a :code:`ParticleDat` labeled by :code:`V`. Finally a value is reduced into a :code:`ScalarArray` labeled :code:`K`. To access data in a :code:`ParticleDat` within the kernel we take our chosen name, for example :code:`M`, and append :code:`.i` to indicate that this operation concerns the first particle under consideration, to access the second particle under consideration, for example in a :code:`PairLoop`, we append a :code:`.j`. In a :code:`ParticleLoop` we may only consider one particle at a time hence in this example all our data access concerns only one particle by using :code:`M.i, V.i, F.i`. Components of :code:`ParticleDat` data structures are indexed through square brackets.
::

    kernel_code = '''
    const double M_tmp = 1.0 / M.i[0];

    // Update the values in a ParticleDat
    V.i[0] += dht * F.i[0] * M_tmp;
    V.i[1] += dht * F.i[1] * M_tmp;
    V.i[2] += dht * F.i[2] * M_tmp;


To access the ScalarArray data we do not append any :code:`.i` or :code:`.j` suffix and instead access the data as if it were a C array.
::

    // Here we reduce into a ScalarArray (kinetic energy example)
    K[0] += V.i[0]*V.i[0] + V.i[1]*V.i[1] + V.i[2]*V.i[2]
    '''



