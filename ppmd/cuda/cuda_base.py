"""
CUDA equivalent of the "host" module at package level.
"""

# System level imports
import ctypes
import numpy as np
import pycuda.gpuarray as gpuarray

#package level imports
import ppmd.access
import ppmd.host

# cuda imports
import cuda_runtime

###############################################################################
# Get available memory.
###############################################################################
def available_free_memory():
    """
    Get available free memory in bytes.
    :return: Free memory in bytes.
    """

    return cuda_runtime.cuda_mem_get_info()[1]


class Struct(ctypes.Structure):
    _fields_ = (('ptr', ctypes.c_void_p), ('ncomp', ctypes.c_void_p))


def _make_gpu_array(initial_value=None, dtype=None, nrow=None, ncol=None):
    """
    dat initialiser
    """

    # pycuda does not take nicely to negative or zero length arrays
    if nrow is not None:
        nrow=max(nrow, 1)
    ncol=max(ncol, 1)


    if initial_value is not None:
        if type(initial_value) is np.ndarray:
            return _create_from_existing(initial_value, dtype)
        else:
            return _create_from_existing(np.array(list(initial_value),
                                                  dtype=dtype), dtype)
    else:
        return _create_zeros(nrow=nrow, ncol=ncol, dtype=dtype)


def _create_zeros(nrow=None, ncol=None, dtype=ctypes.c_double):
    assert ncol is not None, "Make 1D arrays using ncol not nrow"
    if nrow is not None:
        return gpuarray.zeros([nrow, ncol], dtype=dtype)
    else:
        return gpuarray.zeros(ncol, dtype=dtype)


def _create_from_existing(ndarray=None, dtype=ctypes.c_double):
    if ndarray.dtype != dtype:
        ndarray = ndarray.astype(dtype)
    return gpuarray.to_gpu(ndarray)


###############################################################################
# Array.
###############################################################################
class Array(object):
    """
    Basic dynamic memory array on host, with some methods.
    """
    def __init__(self, initial_value=None, name=None, ncomp=1, dtype=ctypes.c_double):
        """
        Creates scalar with given initial value.
        """
        self._name = name
        self.idtype = dtype
        self._ncomp = ctypes.c_int(0)

        self._dat = _make_gpu_array(initial_value=initial_value,
                                    dtype=dtype,
                                    ncol=ncomp)

        self._ncomp.value = self._dat.shape[0]

        self._version = 0
        self._h_mirror = _ArrayMirror(self)

        self._struct = None
        self._init_struct()

    def _init_struct(self):
         self._struct = type('ArrayT',
                            (ctypes.Structure,),
                            dict(_fields_=(('ptr',
                                            ctypes.c_void_p),
                                           ('ncomp', ctypes.POINTER(ctypes.c_int)))))()

    def __call__(self, mode):
        return self, mode

    def __getitem__(self, key):
        self._h_mirror.copy_from_device()
        return self._h_mirror.mirror.data[key]

    def __setitem__(self, key, value):
        if type(key) is tuple or type(key) is slice:
            if type(key) is slice and key == slice(None):
                self._h_mirror.realloc(self.ncomp)
                self._h_mirror.mirror.data[key] = value
            elif len(key) > 1 and all(k == slice(None) for k in key):
                self._h_mirror.realloc(self.ncomp)
                self._h_mirror.mirror.data[key] = value
            else:
                self._h_mirror.copy_from_device()
                self._h_mirror.mirror.data[key] = value
        else:
            self._h_mirror.copy_from_device()
            self._h_mirror.mirror.data[key] = value


        self._h_mirror.copy_to_device()

    def __repr__(self):
        return str(self.__getitem__(slice(None, None, None)))

    @property
    def struct(self):
        self._struct.ptr = self.ctypes_data
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

    @property
    def ncomp(self):
        return self._ncomp.value

    @property
    def size(self):
        return self._ncomp.value * ctypes.sizeof(self.idtype)

    @property
    def ctypes_data(self):
        return ctypes.cast(self._dat.ptr, ctypes.c_void_p)

    def ctypes_data_access(self, mode=ppmd.access.RW, pair=False):

        #print "pre", mode, self[0]
        if mode is ppmd.access.INC0:
            #print self[0], mode
            self.zero()
        return self.ctypes_data

    def ctypes_data_post(self, mode=ppmd.access.RW):

        #print self[0]
        pass

    def realloc(self, ncomp):
        """
        Re allocate memory for an array.
        """
        assert ncomp > 0, "Zero or negative ncomp passed"

        if ncomp != self._ncomp.value:
            _new = _create_zeros(ncol=ncomp, dtype=self.idtype)

            _new[:self._ncomp.value:] = self._dat[:]
            self._dat = _new
            self._ncomp.value = ncomp

    def realloc_zeros(self, ncomp):
        """
        Re allocate memory for an array without copying existing values.
        """
        assert ncomp > 0, "Zero or negative ncomp passed"

        if ncomp != self._ncomp.value:
            self._dat = _create_zeros(ncol=ncomp, dtype=self.idtype)
        else:
            self.zero()


    def zero(self):
        """
        Set all the values in the array to zero.
        """
        self._dat.fill(np.array([0], dtype=self.dtype))
        #print self[0]

    @property
    def dtype(self):
        return self.idtype

    @property
    def end(self):
        """
        Returns end index of array.
        """
        return self.ncomp - 1

    def sync_from_version(self, array=None):
        """
        Keep this array in sync with another array based on version.
        """
        assert array is not None, "cuda_base:Array.sync_from_version error. " \
                                  "No array passed."

        if self.version < array.version:
            self._dat = _create_from_existing(array.data, array.dtype)

###############################################################################
# Matrix.
###############################################################################

class Matrix(object):
    """
    Basic dynamic memory matrix on host, with some methods.
    """
    def __init__(self, nrow=1, ncol=1, initial_value=None, dtype=ctypes.c_double):

        self.idtype = dtype

        self._ncol = ctypes.c_int(0)
        self._nrow = ctypes.c_int(0)

        self._dat = _make_gpu_array(initial_value=initial_value,
                                    dtype=dtype,
                                    nrow=nrow,
                                    ncol=ncol)
        self._nrow.value = self._dat.shape[0]
        self._ncol.value = self._dat.shape[1]

        self._vid_int = 0

        self._h_mirror = _MatrixMirror(self)
        self._struct = None
        self._init_struct()


    def _init_struct(self):
        self._struct = type('MatrixT',
                            (ctypes.Structure,),
                            dict(_fields_=(('ptr',
                            ctypes.c_void_p),
                            ('nrow', ctypes.POINTER(ctypes.c_int)),
                            ('ncol', ctypes.POINTER(ctypes.c_int)))))()

    @property
    def struct(self):
        self._struct.ptr = self.ctypes_data
        self._struct.nrow = ctypes.pointer(self._nrow)
        self._struct.ncol = ctypes.pointer(self._ncol)
        return self._struct

    def __call__(self, mode):
        return self, mode

    @property
    def version(self):
        """
        Get the version of this array.
        :return int version:
        """
        return self._vid_int

    def inc_version(self, inc=1):
        """
        Increment the version by the specified amount
        :param int inc: amount to increment version by.
        """
        self._vid_int += int(inc)

    def realloc(self, nrow=None, ncol=None):
        """
        Re allocate memory for a matrix.
        """
        if (nrow != self._nrow.value) or (ncol != self._ncol.value):
            _new = _create_zeros(nrow=nrow, ncol=ncol, dtype=self.dtype)
            _new[:self._nrow.value:, :self._ncol.value:] = self._dat[:,:]
            self._dat = _new

        self._ncol.value = ncol
        self._nrow.value = nrow

    @property
    def nrow(self):
        return self._nrow.value

    @property
    def ncol(self):
        return self._ncol.value

    @property
    def size(self):
        return self._nrow.value * self._ncol.value * ctypes.sizeof(self.dtype)

    @property
    def ctypes_data(self):
        return ctypes.cast(self._dat.ptr, ctypes.c_void_p)

    def ctypes_data_access(self, mode=ppmd.access.RW):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """
        return self.ctypes_data

    def ctypes_data_post(self, mode=ppmd.access.RW):
        pass

    def zero(self):
        """
        Set all the values in the array to zero.
        """
        self._dat.fill(np.array([0], dtype=self.dtype))

    @property
    def dtype(self):
        return self.idtype

    def __getitem__(self, key):
        self._h_mirror.copy_from_device()
        return self._h_mirror.mirror.data[key]

    def __setitem__(self, key, value):

        self._h_mirror.copy_from_device()
        self._h_mirror.mirror.data[key] = value

        self._h_mirror.copy_to_device()
        self._vid_int += 1

    def __repr__(self):
        return str(self.__getitem__(slice(None, None, None)))

    def sync_from_version(self, matrix=None):
        """
        Keep this array in sync with another matrix based on version.
        """
        assert matrix is not None, "cuda_base:Matrix.sync_from_version error." \
                                   " No matrix passed."

        if self.version < matrix.version:
            self._dat = _create_from_existing(matrix.data, matrix.dtype)



class _ArrayMirror(object):
    def __init__(self, d_array):
        self._d_array = d_array
        self._h_array = ppmd.host.Array(ncomp=d_array.ncomp,
                                        dtype=d_array.dtype)

    def realloc(self, len):
        self._h_array.realloc(len)


    def copy_to_device(self):
        self._d_array.realloc(self._h_array.ncomp)
        cuda_runtime.cuda_mem_cpy(self._d_array.ctypes_data,
                                  self._h_array.ctypes_data,
                                  ctypes.c_size_t(self._h_array.size),
                                  'cudaMemcpyHostToDevice')

    def copy_from_device(self):
        self._h_array.realloc(self._d_array.ncomp)
        cuda_runtime.cuda_mem_cpy(self._h_array.ctypes_data,
                                  self._d_array.ctypes_data,
                                  ctypes.c_size_t(self._h_array.size),
                                  'cudaMemcpyDeviceToHost')

    @property
    def mirror(self):
        return self._h_array


class _MatrixMirror(object):
    def __init__(self, d_matrix):
        self._d_matrix = d_matrix

        self._h_matrix = ppmd.host.Matrix(nrow=1,
                                          ncol=d_matrix.ncol,
                                          dtype=d_matrix.dtype)



    def copy_to_device(self):
        self._d_matrix.realloc(self._h_matrix.nrow, self._h_matrix.ncol)
        cuda_runtime.cuda_mem_cpy(self._d_matrix.ctypes_data,
                                  self._h_matrix.ctypes_data,
                                  ctypes.c_size_t(self._h_matrix.size),
                                  'cudaMemcpyHostToDevice')
        self._h_matrix.data.fill(0)

    def copy_from_device(self):

        self._h_matrix.realloc(self._d_matrix.nrow, self._d_matrix.ncol)
        cuda_runtime.cuda_mem_cpy(self._h_matrix.ctypes_data,
                                  self._d_matrix.ctypes_data,
                                  ctypes.c_size_t(self._h_matrix.size),
                                  'cudaMemcpyDeviceToHost')

    @property
    def mirror(self):
        return self._h_matrix



class device_buffer_2d(object):
    def __init__(self, nrow=1, ncol=1, dtype=ctypes.c_double):
        """
        This is a low level data structure for passing around a 2D cuda buffer.
        The *only* purpose of this should be for libraries which may need to
        realloc memory part way through an expensive operation. e.g. neighbour
        matrix building.
        """

        self._nrow = ctypes.c_int(0)
        self._ncol = ctypes.c_int(0)

        self._ncol.value = 0
        self._nrow.value = 0
        self._alloced = False
        self.dtype = dtype
        self.ctypes_data = ctypes.c_void_p()
        self.realloc(nrow=nrow, ncol=ncol)
        self._struct = type('MatrixT',
                            (ctypes.Structure,),
                            dict(_fields_=(('ptr',
                            ctypes.c_void_p),
                            ('nrow', ctypes.POINTER(ctypes.c_int)),
                            ('ncol', ctypes.POINTER(ctypes.c_int)))))()

    @property
    def nrow(self):
        return self._nrow.value

    @property
    def ncol(self):
        return self._ncol.value

    @nrow.setter
    def nrow(self, val):
        self._nrow.value = val

    @ncol.setter
    def ncol(self, val):
        self._ncol.value = val

    def realloc(self, nrow=1, ncol=1):
        """
        Re allocate memory for a matrix. This will *not* copy th existing
        contents and essentially is for initialisation purposes.
        """
        self.free()
        cuda_runtime.cuda_malloc(self.ctypes_data,
                                 nrow*ncol,
                                 self.dtype)

        self.ncol = ncol
        self.nrow = nrow
        self._alloced = True

    def free(self):
        """
        Free the pointer currently held
        """
        if self._alloced:
            cuda_runtime.cuda_free(self.ctypes_data)
            self._alloced = False

    @property
    def struct(self):
        self._struct.ptr = self.ctypes_data
        self._struct.nrow = ctypes.pointer(self._nrow)
        self._struct.ncol = ctypes.pointer(self._ncol)
        return self._struct







































