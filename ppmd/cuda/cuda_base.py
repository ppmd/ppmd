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
        self._ncomp = 0
        self._ptr = ctypes.POINTER(self.idtype)()


        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)



    def _create_zeros(self, length=1, dtype=ctypes.c_double):
        if dtype != self.dtype:
            self.idtype = dtype

        self.realloc(length)
        self.zero()

    def _create_from_existing(self, ndarray=None, dtype=ctypes.c_double):

        self.idtype = dtype
        if dtype != self.dtype:
            self.idtype = dtype

        self.realloc(ndarray.shape[0])
        cuda_runtime.cuda_mem_cpy(self._ptr,
                                  ndarray.ctypes.data_as(ctypes.POINTER(dtype)),
                                  ctypes.c_size_t(ndarray.shape[0] * ctypes.sizeof(dtype)),
                                  'cudaMemcpyHostToDevice')


    @property
    def ncomp(self):
        return self._ncomp

    @property
    def size(self):
        return self._ncomp * ctypes.sizeof(self.idtype)

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

        if (self._ncomp != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)

        if length != self._ncomp:
            cuda_runtime.cuda_malloc(self._ptr, length, self.idtype)

        self._ncomp = length

    def zero(self):
        """
        Set all the values in the array to zero.
        """
        assert self._ptr is not None, "cuda_base:zero error: pointer type unknown."
        #assert self._ncomp != 0, "cuda_base:zero error: length unknown."

        cuda_runtime.libcudart('cudaMemset', self._ptr, ctypes.c_int(0), ctypes.c_size_t(self._ncomp))

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
        if (self._ncomp != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)




################################################################################################
# Matrix.
################################################################################################

class Matrix(object):
    """
    Basic dynamic memory matrix on host, with some methods.
    """
    def __init__(self, nrow=0, ncol=0, initial_value=None, dtype=ctypes.c_double):

        self.idtype = dtype
        self._ncol = 0
        self._nrow = 0

        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)

        self._cuda_dat = None

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

        if (self._ncol != 0) and (self._nrow != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)


        if (nrow != self._nrow) and (ncol != self._ncol):
            cuda_runtime.cuda_malloc(self._ptr, nrow * ncol, self.idtype)

        self._ncol = ncol
        self._nrow = nrow

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncol(self):
        return self._ncol

    @property
    def size(self):
        return self._nrow * self._ncol * ctypes.sizeof(self.dtype)

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

        cuda_runtime.libcudart('cudaMemset', self._ptr, ctypes.c_int(0), ctypes.c_size_t(self._ncol * self._nrow))

    @property
    def dtype(self):
        return self.idtype

    def free(self):
        if (self._ncol != 0) and (self._nrow != 0) and (self._ptr is not None):
            cuda_runtime.cuda_free(self._ptr)












