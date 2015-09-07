import host
import ctypes
import numpy as np
import datetime
import os
import re
import sys
import math
import pio
import access

np.set_printoptions(threshold='nan')



#####################################################################################
# Scalar array.
#####################################################################################


class ScalarArray(host.Array):
    """
    Base class to hold a single floating point property.
    
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    
    """

    def __init__(self, initial_value=None, name=None, ncomp=1, dtype=ctypes.c_double):
        """
        Creates scalar with given initial value.
        """

        self.name = name
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

    def resize(self, new_length):
        if new_length > self.ncomp:
            self.realloc(new_length)


    def scale(self, val):
        """
        Scale data array by value val.

        :arg double val: Coefficient to scale all elements by.
        """
        self.dat = self.dat * np.array([val], dtype=self.dtype)

    @property
    def ctypes_value(self):
        """Return first value in correct type."""
        return self.dtype(self.dat[0])

    @property
    def min(self):
        """Return minimum"""
        return self.dat.min()

    @property
    def max(self):
        """Return maximum"""
        return self.dat.max()

    @property
    def mean(self):
        """Return mean"""
        return self.dat.mean()

    @property
    def sum(self):
        """
        Return array sum
        """
        return self.dat.sum()

    @property
    def average(self):
        """Returns averages of recorded values since AverageReset was called."""
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
NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)
































