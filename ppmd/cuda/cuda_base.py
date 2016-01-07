"""
CUDA equivalent of the "host" module at package level.
"""

# System level imports
import ctypes
import numpy as np

#package level imports
import ppmd.access

# cuda imports
import cuda_runtime

################################################################################################
# Get available memory.
################################################################################################
def available_free_memory():
    """
    Get available free memory in bytes.
    :return: Free memory in bytes.
    """

    return cuda_runtime.cuda_mem_get_info()[1]


class Struct(ctypes.Structure):
    _fields_ = (('ptr', ctypes.c_void_p), ('ncomp', ctypes.c_void_p))

################################################################################################
# Array.
################################################################################################
class Array(object):
    """
    Basic dynamic memory array on host, with some methods.
    """
    def __init__(self, initial_value=None, ncomp=0, dtype=ctypes.c_double):
        """
        Creates scalar with given initial value.
        """
        self.idtype = dtype
        self._ncomp = ctypes.c_int(0)

        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)

        self._struct = type('ArrayT', (ctypes.Structure,), dict(_fields_=(('ptr', ctypes.POINTER(self.idtype)), ('ncomp', ctypes.POINTER(ctypes.c_int)))))()


        self._version = 0

    @property
    def struct(self):
        self._struct.ptr = self._ptr
        self._struct.ncomp = ctypes.pointer(self._ncomp)
        return self._struct


    @property
    def version(self):
        """
        Get the version of this array.
        :return int version:
        """
        return self._version

    def inc_version(self, inc=1):
        """
        Increment the version by the specified amount
        :param int inc: amount to increment version by.
        """
        self._version += int(inc)

    def _create_zeros(self, length=1, dtype=ctypes.c_double):
        if dtype != self.dtype:
            self.idtype = dtype

        if length > 0:
            self.realloc(length)
            self.zero()

    def _create_from_existing(self, ndarray=None, dtype=ctypes.c_double):

        if dtype != ndarray.dtype:
            print "cuda_base:Array._create_from_existing() data type miss matched."

        self.idtype = dtype
        if dtype != self.dtype:
            self.idtype = dtype

        self.realloc(ndarray.size)
        cuda_runtime.cuda_mem_cpy(self._ptr,
                                  ndarray.ctypes.data_as(ctypes.POINTER(dtype)),
                                  ctypes.c_size_t(ndarray.size * ctypes.sizeof(dtype)),
                                  'cudaMemcpyHostToDevice')


    @property
    def ncomp(self):
        return self._ncomp.value

    @property
    def size(self):
        return self._ncomp.value * ctypes.sizeof(self.idtype)

    @property
    def ctypes_data(self):
        return self._ptr

    def ctypes_data_access(self, mode=ppmd.access.RW):
        return self._ptr

    def ctypes_data_post(self, mode=ppmd.access.RW):
        pass

    def realloc(self, length):
        """
        Re allocate memory for an array.
        """
        assert self._ptr is not None, "cuda_base.Array: realloc error: pointer type unknown."

        if (self._ncomp.value != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)

        if length != self._ncomp.value:
            cuda_runtime.cuda_malloc(self._ptr, length, self.idtype)

        self._ncomp.value = length

    def zero(self):
        """
        Set all the values in the array to zero.
        """
        assert self._ptr is not None, "cuda_base:zero error: pointer type unknown."
        #assert self._ncomp != 0, "cuda_base:zero error: length unknown."

        cuda_runtime.libcudart('cudaMemset', self._ptr, ctypes.c_int(0), ctypes.c_size_t(self._ncomp.value * ctypes.sizeof(self.idtype)))

    @property
    def dtype(self):
        return self.idtype

    @property
    def end(self):
        """
        Returns end index of array.
        """
        return self.ncomp - 1

    def free(self):
        if (self._ncomp.value != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)


    def sync_from_version(self, array=None):
        """
        Keep this array in sync with another array based on version.
        """
        assert array is not None, "cuda_base:Array.sync_from_version error. No array passed."

        if self.version < array.version:

            self._create_from_existing(array.dat, array.dtype)










################################################################################################
# Matrix.
################################################################################################

class Matrix(object):
    """
    Basic dynamic memory matrix on host, with some methods.
    """
    def __init__(self, nrow=0, ncol=0, initial_value=None, dtype=ctypes.c_double):

        self.idtype = dtype


        self._ncol = ctypes.c_int(0)
        self._nrow = ctypes.c_int(0)

        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)

        self._version = 0

        self._struct = type('MatrixT', (ctypes.Structure,), dict(_fields_=(('ptr', ctypes.POINTER(self.idtype)),
                                                                          ('nrow', ctypes.POINTER(ctypes.c_int)),
                                                                          ('ncol', ctypes.POINTER(ctypes.c_int)))))()


    @property
    def struct(self):
        self._struct.ptr = self._ptr
        self._struct.nrow = ctypes.pointer(self._nrow)
        self._struct.ncol = ctypes.pointer(self._ncol)
        return self._struct

    @property
    def version(self):
        """
        Get the version of this array.
        :return int version:
        """
        return self._version

    def inc_version(self, inc=1):
        """
        Increment the version by the specified amount
        :param int inc: amount to increment version by.
        """
        self._version += int(inc)

    def _create_zeros(self, nrow=1, ncol=1, dtype=ctypes.c_double):
        if dtype != self.dtype:
            self.idtype = dtype

        self.realloc(nrow, ncol)
        self.zero()

    def _create_from_existing(self, ndarray=None, dtype=ctypes.c_double):

        self.idtype = dtype
        if dtype != self.dtype:
            self.idtype = dtype

        self.realloc(ndarray.shape[0], ndarray.shape[1])
        cuda_runtime.cuda_mem_cpy(self._ptr,
                                  ndarray.ctypes.data_as(ctypes.POINTER(dtype)),
                                  ctypes.c_size_t(ndarray.shape[0] * ndarray.shape[1] * ctypes.sizeof(dtype)),
                                  'cudaMemcpyHostToDevice')

    def realloc(self, nrow=None, ncol=None):
        """
        Re allocate memory for a matrix.
        """
        assert self._ptr is not None, "cuda_base.Matrix: realloc error: pointer type unknown."

        if (self._ncol.value != 0) and (self._nrow.value != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)


        if (nrow != self._nrow.value) and (ncol != self._ncol.value):
            cuda_runtime.cuda_malloc(self._ptr, nrow * ncol, self.idtype)

        self._ncol.value = ncol
        self._nrow.value = nrow

    @property
    def nrow(self):
        return self._nrow.value
    
    @nrow.setter
    def nrow(self, val):
        self._nrow.value = val
        if cuda_runtime.VERBOSE > 2:
            print "cuda_base.Matrix warning: nrow externally changed."

    @property
    def ncol(self):
        return self._ncol.value

    @ncol.setter
    def ncol(self, val):
        self._ncol.value = val
        if cuda_runtime.VERBOSE > 2:
            print "cuda_base.Matrix warning: ncol externally changed."

    @property
    def size(self):
        return self._nrow.value * self._ncol.value * ctypes.sizeof(self.dtype)

    @property
    def ctypes_data(self):
        return self._ptr

    def ctypes_data_access(self, mode=ppmd.access.RW):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """
        return self._ptr

    def ctypes_data_post(self, mode=ppmd.access.RW):
        pass

    def zero(self):
        """
        Set all the values in the matrix to zero.
        """
        assert self._ptr is not None, "cuda_base.Matrix: zero error: pointer type unknown."
        #assert self._ncomp != 0, "cuda_base:zero error: length unknown."

        cuda_runtime.libcudart('cudaMemset', self._ptr, ctypes.c_int(0), ctypes.c_size_t(self._ncol.value * self._nrow.value * ctypes.sizeof(self.idtype)))

    @property
    def dtype(self):
        return self.idtype

    def free(self):
        if (self._ncol.value != 0) and (self._nrow.value != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)


    def sync_from_version(self, matrix=None):
        """
        Keep this array in sync with another array based on version.
        """
        assert matrix is not None, "cuda_base:Array.sync_from_version error. No array passed."

        if self.version < matrix.version:

            self._create_from_existing(matrix.dat, matrix.dtype)









