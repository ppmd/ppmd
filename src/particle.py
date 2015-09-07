import numpy as np
import ctypes
import access
import halo
import host


class Dat(host.Matrix):
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
