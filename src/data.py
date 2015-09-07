import host
import ctypes
import numpy as np
import access
import halo
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


class ParticleDat(host.Matrix):
    """
    Base class to hold floating point properties of particles, creates N1*N2 array with given initial value.

    :arg int npart: First dimension extent (number of particles).
    :arg int ncol: Second dimension extent (number of components).
    :arg double initial_value: Value to initialise array with, default zeros.
    :arg str name: Collective name of stored vars eg positions.

    """

    def __init__(self, npart=1, ncomp=1, initial_value=None, name=None, dtype=ctypes.c_double, max_npart=None):

        self.name = name
        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

            self.max_npart = self.nrow
            self.npart = self.nrow
            self.ncomp = self.ncol

        else:
            if max_npart is None:
                max_npart = npart
            self._create_zeros(max_npart, ncomp, dtype)
            self.max_npart = self.nrow
            self.npart = self.nrow
            self.ncomp = self.ncol

        self.halo_start = self.npart

        '''Number of halo particles'''
        self.npart_halo = 0


    def set_val(self, val):
        self.dat[..., ...] = val

    def __getitem__(self, ix):
        return self.dat[ix]

    def __setitem__(self, ix, val):
        self.dat[ix] = val

    def __str__(self):
        return str(self._Dat[::])

    def __call__(self, mode=access.RW, halo=True):

        return self, mode


    def halo_start_shift(self, shift):
        self.halo_start += shift
        self.npart_halo = self.halo_start - self.npart

    def halo_start_set(self, index):

        if index < self.npart:
            if index >= 0:
                self.npart = index
                self.halo_start = index

        else:
            self.halo_start_reset()

        self.npart_halo = 0

    def halo_start_reset(self):
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
        halo.HALOS.exchange(self)




class TypedDat(host.Matrix):
    """
    Base class to hold floating point properties of particles based on particle type, creates N1*N2 array with given initial value.

    :arg int nrow: First dimension extent.
    :arg int ncol: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.

    """

    def __init__(self, nrow=1, ncol=1, initial_value=None, name=None, dtype=ctypes.c_double):

        self.name = str(name)
        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)





























