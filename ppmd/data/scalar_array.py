from __future__ import print_function, division, absolute_import
"""
This module contains high level arrays and matrices.
"""

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np

# package level
from ppmd import access, mpi, host

np.set_printoptions(threshold=1000)

_MPI = mpi.MPI
SUM = _MPI.SUM
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier


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

        self.halo_aware = False
        """How to handle writes to this dat in a reduction sense. """ \
        """In general for a reduction in a pair loop a write will occur once per pair """ \
        """In the case where one of the pair is in a halo, the write will occur if the ith""" \
        """particle is not in the halo?"""

        self.name = name
        """Name of ScalarArray instance."""

        self.idtype = dtype
        self.data = host._make_array(initial_value=initial_value,
                                     dtype=dtype,
                                     ncol=ncomp)

        self._version = 0

    def __call__(self, mode=access.RW, halo=True):
        return self, mode

    def __setitem__(self, ix, val):
        self.data[ix] = np.array([val], dtype=self.dtype)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

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
        self.data = self.data * np.array([val], dtype=self.dtype)

    @property
    def ctypes_value(self):
        """:return: first value in correct type."""
        return self.dtype(self.data[0])

    @property
    def min(self):
        """:return: The minimum in the array."""
        return self.data.min()

    @property
    def max(self):
        """:return: The maximum value in the array."""
        return self.data.max()

    @property
    def mean(self):
        """:return: The mean value in the array."""
        return self.data.mean()

    @property
    def sum(self):
        """
        :return: The array sum.
        """
        return self.data.sum()

