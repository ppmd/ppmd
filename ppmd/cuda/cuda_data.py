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

    def __init__(self, initial_value=None , name=None, ncomp=0, dtype=ctypes.c_double):

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




















