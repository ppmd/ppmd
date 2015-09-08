"""
This module contains high level arrays and matrices.
"""


import host
import ctypes
import numpy as np
import access
import halo
np.set_printoptions(threshold=1000)


"""
rst_doc{

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


}rst_doc
"""


#####################################################################################
# Scalar array.
#####################################################################################


class ScalarArray(host.Array):
    """
    Class to hold an array of scalar values.
    
    :arg initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    :arg dtype: Data type. Should be a ctypes C data type.
    """

    def __init__(self, initial_value=None, name=None, ncomp=1, dtype=ctypes.c_double):
        """
        Creates scalar with given initial value.
        """

        self.name = name
        """Name of ScalarArray instance."""

        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)

        self._A = False
        self._Aarray = None

    def __call__(self, access=access.RW, halo=True):
        return self

    def __setitem__(self, ix, val):
        self.dat[ix] = np.array([val], dtype=self.dtype)

        if self._A is True:
            self._Aarray[ix] = np.array([val], dtype=self.dtype)
            self._Alength += 1

    def __str__(self):
        return str(self.dat)

    def __repr__(self):
        return str(self.dat)

    def resize(self, new_length):
        """
        Increase the size of the array.
        :param int new_length: New array length.
        """
        if new_length > self.ncomp:
            self.realloc(new_length)

    def scale(self, val):
        """
        Scale data array by given value.

        :arg val: Coefficient to scale all elements by.
        """
        self.dat = self.dat * np.array([val], dtype=self.dtype)

    @property
    def ctypes_value(self):
        """:return: first value in correct type."""
        return self.dtype(self.dat[0])

    @property
    def min(self):
        """:return: The minimum in the array."""
        return self.dat.min()

    @property
    def max(self):
        """:return: The maximum value in the array."""
        return self.dat.max()

    @property
    def mean(self):
        """:return: The mean value in the array."""
        return self.dat.mean()

    @property
    def sum(self):
        """
        :return: The array sum.
        """
        return self.dat.sum()

    @property
    def average(self):
        """:return: averages of recorded values since AverageReset was called."""
        # assert self._A == True, "Run AverageReset to initialise or reset averaging"
        if self._A is True:
            return self._Aarray / self._Alength

    def average_stop(self, clean=False):
        """
        Stops averaging values.
        
        :arg bool clean: Flag to free memory allocated to averaging, default False.
        """
        if self._A is True:
            self._A = False
            if clean is True:
                del self._A

    def average_update(self):
        """Copy values from Dat into averaging array"""
        if self._A is True:
            self._Aarray += self.dat
            self._Alength += 1
        else:
            self.average_reset()
            self._Aarray += self.dat
            self._Alength += 1


###################################################################################################
# Blank arrays.
###################################################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
"""Empty integer :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""


NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)
"""Empty double :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""

###################################################################################################
# ParticleDat.
###################################################################################################


class ParticleDat(host.Matrix):
    """
    Base class to hold properties of particles. This could be considered as a two dimensional matrix
    with each row representing the stored properties of a particle.

    :arg int npart: Number of particles (Number of row in matrix).
    :arg int ncol: Dimension of property to store per particle (Number of columns in matrix).
    :arg initial_value: Value to initialise array with, default zeros.
    :arg str name: Collective name of stored vars eg positions.
    """

    def __init__(self, npart=1, ncomp=1, initial_value=None, name=None, dtype=ctypes.c_double, max_npart=None):

        self.name = name
        """:return: The name of the ParticleDat instance."""

        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)

            self.max_npart = self.nrow
            """:return: The maximum number of particles which can be stored within this particle dat."""

            self.npart = self.nrow
            """:return: The number of particles with properties stored in the particle dat."""

            self.ncomp = self.ncol
            """:return: The number of components stored for each particle."""

        else:
            if max_npart is None:
                max_npart = npart
            self._create_zeros(max_npart, ncomp, dtype)
            self.max_npart = self.nrow
            self.npart = self.nrow
            self.ncomp = self.ncol

        self.halo_start = self.npart
        """:return: The starting index of the halo region of the particle dat. """

        self.npart_halo = 0
        """:return: The number of particles currently stored within the halo region of the particle dat."""

    def set_val(self, val):
        """
        Set all the entries in the particle dat to the same specified value.

        :param val: Value to set all entries to.
        """
        self.dat[..., ...] = val

    def __getitem__(self, ix):
        return self.dat[ix]

    def __setitem__(self, ix, val):
        self.dat[ix] = val

    def __str__(self):
        return str(self._Dat[::])

    def __repr__(self):
        self.__str__()

    def __call__(self, mode=access.RW, halo=True):

        return self, mode

    def halo_start_shift(self, shift):
        """
        Shift the starting point of the halo in the particle dat by the specified shift.
        :param int shift: Offset to shift by.
        """

        self.halo_start += shift
        self.npart_halo = self.halo_start - self.npart

    def halo_start_set(self, index):
        """
        Set the start of the halo region in the particle dat to the specified index.
        :param int index: Index to set to.
        """
        if index < self.npart:
            if index >= 0:
                self.npart = index
                self.halo_start = index

        else:
            self.halo_start_reset()

        self.npart_halo = 0

    def halo_start_reset(self):
        """
        Reset the starting postion of the halo region in the particle dat to the end of the
        local particles.
        :return:
        """
        self.halo_start = self.npart
        self.npart_halo = 0

    def resize(self, n):
        """
        Resize particle dat to be at least a certain size, does not resize if already large enough.
        :arg int n: New minimum size.
        """

        if n > self.max_npart:
            self.max_npart = n
            self.realloc(n, self.ncol)

    def halo_exchange(self):
        """
        Perform a halo exchange for the particle dat. WIP currently only functional for positions.
        """
        halo.HALOS.exchange(self)

###################################################################################################
# ParticleDat.
###################################################################################################


class TypedDat(host.Matrix):
    """
    Base class to hold floating point properties in matrix form of particles based on particle type.

    :arg int nrow: First dimension extent.
    :arg int ncol: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    """

    def __init__(self, nrow=1, ncol=1, initial_value=None, name=None, dtype=ctypes.c_double):

        self.name = str(name)
        """:return: Name of TypedDat instance."""
        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)





























