

.. contents:

data Module
===========

.. automodule:: data

Scalar Array
~~~~~~~~~~~~

The class :class:`~data.ScalarArray` is a generic one dimensional array that should be used
to store data within simulations that is not associated with any particular particles. For
example the kinetic energy of the system or the array used to bin values when calculating a
radial distribution.

.. autoclass:: data.ScalarArray
    :show-inheritance:
    :undoc-members:
    :members:


Particle Dat
~~~~~~~~~~~~

This classes should be considered as a two dimensional matrix with each row storing the properties
of a particle. The order of rows in relation to which particle they correspond to should always
be conserved. This is the default behaviour of any sorting methods implemented in this framework.

.. autoclass:: data.ParticleDat
    :show-inheritance:
    :undoc-members:
    :members:

Typed Dat
~~~~~~~~~

Instances of this class should be used to store properties of particles which are common to
multiple particles e.g. mass.


.. autoclass:: data.TypedDat
    :show-inheritance:
    :undoc-members:
    :members:


