import ctypes
import particle
import numpy as np

_GRAPHICS = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _GRAPHICS = False

import datetime
import os
import re
import pickle
import sys
import math
import runtime
import pio
import gpucuda
import access

np.set_printoptions(threshold='nan')



#####################################################################################
# Scalar array.
#####################################################################################


class ScalarArray(object):
    """
    Base class to hold a single floating point property.
    
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    
    """

    def __init__(self, initial_value=None, name=None, ncomp=1, val=None, dtype=ctypes.c_double, max_size=None):
        """
        Creates scalar with given initial value.
        """

        if max_size is None:
            self._max_size = ncomp
        else:
            self._max_size = max_size

        self._dtype = dtype

        if name is not None:
            self._name = name
        self._N1 = ncomp

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._Dat = np.array(initial_value, dtype=self._dtype, order='C')
                self._N1 = initial_value.shape[0]
                self._max_size = self._N1
            elif type(initial_value) == list:
                self._Dat = np.array(np.array(initial_value), dtype=self._dtype, order='C')
                self._N1 = len(initial_value)
                self._max_size = self._N1
            else:
                self._Dat = float(initial_value) * np.ones([self._N1], dtype=self._dtype, order='C')

        elif val is None:
            self._Dat = np.zeros([self._max_size], dtype=self._dtype, order='C')
        elif val is not None:
            self._Dat = np.array([val], dtype=self._dtype, order='C')

        self._A = False
        self._Aarray = None
        self._DatHaloInit = False
        self._cuda_dat = None


    @property
    def dat(self):
        """
        Returns stored data as numpy array.
        """
        return self._Dat

    @dat.setter
    def dat(self, val):
        if type(val) is np.ndarray:
            self._Dat = val
        else:
            self._Dat = np.array([val], dtype=self._dtype)


    def __getitem__(self, ix):
        return self._Dat[ix]

    def scale(self, val):
        """
        Scale data array by value val.
        
        :arg double val: Coefficient to scale all elements by.
        """

        self._Dat = self._Dat * np.array([val], dtype=self._dtype)

    def __call__(self, access=access.RW, halo=True):

        return self

    def zero(self):
        """
        Zero all elements in array.
        """

        self._Dat = np.zeros(self._N1, dtype=self._dtype, order='C')

    def __setitem__(self, ix, val):
        self._Dat[ix] = np.array([val], dtype=self._dtype)

        if self._A is True:
            self._Aarray[ix] = np.array([val], dtype=self._dtype)
            self._Alength += 1

    def __str__(self):
        return str(self._Dat)

    @property
    def ctypes_data(self):
        """Return ctypes-pointer to data."""
        return self._Dat.ctypes.data_as(ctypes.POINTER(self._dtype))

    @property
    def dtype(self):
        """ Return Dat c data ctype"""
        return self._dtype

    @property
    def ctypes_value(self):
        """Return first value in correct type."""
        return self._dtype(self._Dat[0])

    @property
    def name(self):
        """
        Returns name of particle dat.
        """
        return self._name

    @property
    def ncomp(self):
        """
        Return number of components.
        """
        return self._N1

    @property
    def max_size(self):
        """
        Return actual length of array.
        """
        return self._max_size

    @ncomp.setter
    def ncomp(self, val):
        assert val <= self._max_size, "ncomp, max_size error"
        self._N1 = val

    @property
    def min(self):
        """Return minimum"""
        return self._Dat.min()

    @property
    def max(self):
        """Return maximum"""
        return self._Dat.max()

    @property
    def mean(self):
        """Return mean"""
        return self._Dat.mean()

    @property
    def name(self):
        return self._name

    def resize(self, n):
        if n > self._max_size:
            self._max_size = n + (n - self._max_size) * 10
            self._Dat = np.resize(self._Dat, self._max_size)
            # self._N1 = n

    @property
    def end(self):
        """
        Returns end index of array.
        """
        return self._max_size - 1

    @property
    def sum(self):
        """
        Return array sum
        """
        return self._Dat.sum()

    def dat_write(self, dir_name='./output', filename=None, rename_override=False):
        """
        Function to write ScalarArray objects to disk.
        
        :arg str dir_name: directory to write to, default ./output.
        :arg str filename: Filename to write to, default array name or data.SArray if name unset.
        :arg bool rename_override: Flagging as True will disable autorenaming of output file.
        """

        if (self._name is not None) and (filename is None):
            filename = str(self._name) + '.SArray'
        if filename is None:
            filename = 'data.SArray'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if os.path.exists(os.path.join(dir_name, filename)) & (rename_override is not True):
            filename = re.sub('.SArray', datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.SArray', filename)
            if os.path.exists(os.path.join(dir_name, filename)):
                filename = re.sub('.SArray', datetime.datetime.now().strftime("_%f") + '.SArray', filename)
                assert os.path.exists(os.path.join(dir_name, filename)), "DatWrite Error: No unquie name found."

        f = open(os.path.join(dir_name, filename), 'w')
        pickle.dump(self._Dat, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def dat_read(self, dir_name='./output', filename=None):
        """
        Function to read Dat objects from disk.
        
        :arg str dir_name: directory to read from, default ./output.
        :arg str filename: filename to read from.
        """

        assert os.path.exists(dir_name), "Read directory not found"
        assert filename is not None, "DatRead Error: No filename given."

        f = open(os.path.join(dir_name, filename), 'r')
        self = pickle.load(f)
        f.close()

    def average_reset(self):
        """Reset and initialises averaging."""
        if self._A is False:

            self._Aarray = np.zeros([self._N1], dtype=self._dtype, order='C')
            self._Alength = 0.0
            self._A = True
        else:
            self._Aarray.fill(0.)
            self._Alength = 0.0

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
            self._Aarray += self._Dat
            self._Alength += 1
        else:
            self.average_reset()
            self._Aarray += self._Dat
            self._Alength += 1

    def init_halo_dat(self):
        """
        Create a secondary dat container.
        """

        if self._DatHaloInit is False:
            self._max_size *= 2
            self._Dat = np.resize(self._Dat, self._max_size)
            self._DatHaloInit = True

    @property
    def dat_halo_init(self):
        """
        Return status of halo dat.
        """
        return self._DatHaloInit

    def add_cuda_dat(self):
        """
        Create a corresponding CudaDeviceDat.
        """
        if self._cuda_dat is None:
            self._cuda_dat = gpucuda.CudaDeviceDat(size=self._max_size, dtype=self._dtype)

    def get_cuda_dat(self):
        """
        Returns associated cuda dat, or None if not initialised.
        """
        return self._cuda_dat

    def copy_to_cuda_dat(self):
        """
        Copy the CPU dat to the cuda device.
        """
        assert self._cuda_dat is not None, "particle.dat error: cuda_dat not created."
        self._cuda_dat.cpy_htd(self.ctypes_data)

    def copy_from_cuda_dat(self):
        """
        Copy the device dat into the cuda dat.
        """
        assert self._cuda_dat is not None, "particle.dat error: cuda_dat not created."
        self._cuda_dat.cpy_dth(self.ctypes_data)



###################################################################################################
# Blank arrays.
###################################################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)
































