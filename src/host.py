import ctypes
import numpy as np

ctypes_map = {ctypes.c_double: 'double', ctypes.c_int: 'int', 'float64': 'double', 'int32': 'int',
              'doublepointerpointer': 'double **', ctypes.c_longlong: 'long long'}


################################################################################################
# Array.
################################################################################################

class Array(object):
    """
    Basic dynamic memory array on host, with some methods.
    """
    def _create(self, length=1, dtype=ctypes.c_double):
        self.dat = np.zeros(length, dtype=dtype, order='C')

    @property
    def size(self):
        return self.dat.shape[0] * ctypes.sizeof(self.dat.dtype)

    @property
    def ctypes_data(self):
        return self.dat.ctypes.data_as(ctypes.POINTER(self.dat.dtype))

    def realloc(self, length):
        self.dat.resize(length)

    def zero(self):
        self.dat.fill(self.dat.dtype(0.))

    @property
    def dtype(self):
        return self.dat.dtype


################################################################################################
# Pointer array.
################################################################################################

class PointerArray(object):
    """
    Class to store arrays of pointers.

    :arg int length: Length of array.
    :arg ctypes.dtype dtype: pointer data type.
    """

    def __init__(self, length, dtype):
        self._length = length
        self._dtype = dtype
        self._Dat = (ctypes.POINTER(self._dtype) * self._length)()

    @property
    def dtype(self):
        """Returns data type."""
        return self._dtype

    @property
    def ctypes_data(self):
        """Returns pointer to start of array."""
        return self._Dat

    @property
    def ncomp(self):
        """Return number of components"""
        return self._length

    def __getitem__(self, ix):
        return self._Dat[ix]

    def __setitem__(self, ix, val):
        self._Dat[ix] = val
