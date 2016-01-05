"""
CUDA version of the package level data.py
"""
# System level imports
import ctypes
import numpy as np

#package level imports
import ppmd.access

# cuda imports
import cuda_base


class ScalarArray(cuda_base.Array):

    def __init__(self, initial_value=None, name=None, ncomp=0, dtype=ctypes.c_double):

        self.name = name

        self.idtype = dtype
        self._ncomp = 0
        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)

        self._version = 0

    @property
    def version(self):
        """
        Get the version of this array.
        :return int version:
        """
        return self._version

    def resize(self, new_length):
        """
        Increase the size of the array.
        :param int new_length: New array length.
        """
        if new_length > self.ncomp:
            self.realloc(new_length)

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
# cuda ParticleDat
###################################################################################################


class ParticleDat(cuda_base.Matrix):
    """
    Cuda particle dat
    """
    def __init__(self, npart=0, ncomp=0, initial_value=None, name=None, dtype=ctypes.c_double, max_npart=None):

        self.name = name

        self.idtype = dtype
        self._ncol = 0
        self._nrow = 0

        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

            self.max_npart = self._nrow
            self.npart = self._nrow

        else:
            if max_npart is not None:
                self.max_npart = max_npart
            else:
                self.max_npart = npart

            self._create_zeros(self.max_npart, ncomp, dtype)
            self.npart = npart

        self.ncomp = self._ncol

        self._version = 0

        self.halo_start = self.npart
        self.npart_halo = 0

    @property
    def npart_total(self):
        return self.npart + self.npart_halo


    def resize(self, n):
        if n > self.max_npart:
            self.max_npart = n
            self.realloc(n, self.ncol)









