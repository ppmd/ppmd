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
    def __init__(self, initial_value=None, ncomp=1, dtype=ctypes.c_double):
        """
        Creates scalar with given initial value.
        """
        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)


    def _create_zeros(self, length=1, dtype=ctypes.c_double):
        self.idtype = dtype
        self.dat = np.zeros(length, dtype=dtype, order='C')

    def _create_from_existing(self, ndarray=None, dtype=ctypes.c_double):
        self.idtype = dtype
        self.dat = np.array(ndarray, dtype=dtype, order='C')

    @property
    def ncomp(self):
        return self.dat.shape[0]

    @property
    def size(self):
        return self.dat.shape[0] * ctypes.sizeof(self.dtype)

    @property
    def ctypes_data(self):
        return self.dat.ctypes.data_as(ctypes.POINTER(self.dtype))

    def realloc(self, length):
        self.dat.resize(length)

    def zero(self):
        self.dat.fill(self.idtype(0))

    @property
    def dtype(self):
        return self.idtype

    @property
    def end(self):
        """
        Returns end index of array.
        """
        return self.ncomp - 1

    def __getitem__(self, ix):
        return self.dat[ix]

    def __setitem__(self, ix, val):
        self.dat[ix] = val




################################################################################################
# Matrix.
################################################################################################

class Matrix(object):
    """
    Basic dynamic memory matrix on host, with some methods.
    """
    def __init__(self, nrow=1, ncol=1, initial_value=None, dtype=ctypes.c_double):
        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)


    def _create_zeros(self, nrow=1, ncol=1, dtype=ctypes.c_double):
        self.idtype = dtype
        self.dat = np.zeros([nrow, ncol], dtype=dtype, order='C')

    def _create_from_existing(self, ndarray=None, dtype=ctypes.c_double):
        self.idtype = dtype
        self.dat = np.array(ndarray, dtype=dtype, order='C')
        if len(self.dat.shape) == 1:
            self.dat.shape = (self.dat.shape[0], 1)

    @property
    def nrow(self):
        return self.dat.shape[0]

    @property
    def ncol(self):
        return self.dat.shape[1]

    @property
    def size(self):
        return self.dat.shape[0] * self.dat.shape[1] * ctypes.sizeof(self.dtype)

    @property
    def ctypes_data(self):
        return self.dat.ctypes.data_as(ctypes.POINTER(self.dtype))

    def realloc(self, nrow, ncol):
        self.dat.resize(nrow, ncol)

    def zero(self):
        self.dat.fill(self.idtype(0))

    @property
    def dtype(self):
        return self.idtype

###################################################################################################
# Blank arrays.
###################################################################################################

NullIntArray = Array(dtype=ctypes.c_int)
NullDoubleArray = Array(dtype=ctypes.c_double)

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
