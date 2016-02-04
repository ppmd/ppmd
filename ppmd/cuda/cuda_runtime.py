"""
Module to handle the cuda runtime environment.
"""
#system level imports
import ctypes
import os

#package level imports
from ppmd import runtime, pio, mpi


ERROR_LEVEL = runtime.Level(3)
DEBUG = runtime.Level(0)
VERBOSE = runtime.Level(0)
BUILD_TIMER = runtime.Level(0)

BUILD_DIR = runtime.BUILD_DIR

LIB_DIR = runtime.Dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/'))


#cuda level imports
import cuda_build




#####################################################################################
# load libraries
#####################################################################################

try:
    CUDA_INC_PATH = os.environ['CUDA_INSTALL_PATH']
except KeyError:
    if ERROR_LEVEL.level > 2:
        raise RuntimeError('cuda_runtime error: cuda toolkit environment path not found, expecting CUDA_INSTALL_PATH')
    CUDA_INC_PATH = None

try:
    LIBCUDART = ctypes.cdll.LoadLibrary(CUDA_INC_PATH + "/lib64/libcudart.so")

except:
    if ERROR_LEVEL.level > 2:
        raise RuntimeError('cuda_runtime error: Module is not initialised correctly, CUDA runtime not loaded')
    LIBCUDART = None

# wrapper library for functions involving types.

try:
    LIBHELPER = ctypes.cdll.LoadLibrary(cuda_build.build_static_libs('cudaHelperLib'))
except:
    raise RuntimeError('cuda_runtime error: Module is not initialised correctly, CUDA helper lib not loaded')
    LIBHELPER = None

LIBHELPER['cudaErrorCheck'](ctypes.c_int(0))
#####################################################################################
# Device id of currently used device. Assuming model of one mpi process per gpu.
#####################################################################################

class Device(object):
    def __init__(self):
        self._dev_id = None

    @property
    def id(self):
        return self._dev_id

    @id.setter
    def id(self, new_id):
        self._dev_id = int(new_id)

DEVICE = Device()


#####################################################################################
# cuda_err_checking
#####################################################################################

def cuda_err_check(err_code):
    """
    Wrapper to check cuda error codes.
    :param err_code:
    :return:
    """

    assert LIBHELPER is not None, "cuda_runtime error: No error checking library"

    if LIBHELPER is not None:
        LIBHELPER['cudaErrorCheck'](err_code)


#####################################################################################
# CUDA runtime handle
#####################################################################################

def libcudart(*args):
    """
    Wrapper to cuda runtime library with error code checking.
    :param args: string <function name>, args.
    :return:
    """

    assert LIBCUDART is not None, "cuda_runtime error: No CUDA Runtime library loaded"

    if VERBOSE.level > 2:
        pio.rprint(args)

    cuda_err_check(LIBCUDART[args[0]](*args[1::]))


#####################################################################################
# cuda_set_device. Assuming model of one mpi process per gpu.
#####################################################################################

def cuda_set_device(device=None):
    """
    Set the cuda device.
    :param int device: Dev id to use. If not set a device will be chosen based on rank.
    :return:
    """
    if device is None:
        _r = 0

        try:
            _mv2r = os.environ['MV2_COMM_WORLD_LOCAL_RANK']
        except KeyError:
            _mv2r = None

        try:
            _ompr = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        except KeyError:
            _ompr = None

        if (_mv2r is None) and (_ompr is None):
            print("cuda_runtime warning: Did not find local rank, defaulting to device 0")

        if (_mv2r is not None) and (_ompr is not None):
            print("cuda_runtime warning: Found two local ranks, defaulting to device 0")

        if _mv2r is not None:
            _r = int(_mv2r) % mpi.MPI_HANDLE.nproc
        if _ompr is not None:
            _r = int(_ompr) % mpi.MPI_HANDLE.nproc

    else:
        _r = int(device)

    if LIBCUDART is not None:
        if runtime.VERBOSE.level > 0:
            pio.rprint("setting device ", _r)

        LIBCUDART['cudaSetDevice'](ctypes.c_int(_r))
        libcudart('cudaSetDeviceFlags',ctypes.c_uint(8))
        DEVICE.id = _r
    else:
        pio.rprint("cuda_runtime warning: No device set")

    if (runtime.VERBOSE.level > 0) and (LIBCUDART is not None):
        _dev = ctypes.c_int()
        LIBCUDART['cudaGetDevice'](ctypes.byref(_dev))
        pio.rprint("cudaGetDevice returned device ", _dev.value)



#####################################################################################
# cuda_device_reset
#####################################################################################

def cuda_device_reset():
    """
    Reset the current cuda device.
    :return:
    """

    if (DEVICE.id is not None) and (LIBCUDART is not None):
        libcudart('cudaDeviceReset')


#####################################################################################
# Is module ready to use?
#####################################################################################

def INIT_STATUS():
    """
    Function to determine if the module is correctly loaded and can be used.
    :return: True/False.
    """

    if (LIBCUDART is not None) and (LIBHELPER is not None) and (DEVICE.id is not None) and runtime.CUDA_ENABLED.flag:
        return True
    else:
        return False



#####################################################################################
#  cuMemGetInfo
#####################################################################################

def cuda_mem_get_info():
    """
    Get the total memory available and the total free memory.
    :return: Int Tuple (total, free)
    """
    _total = (ctypes.c_size_t * 1)()
    _total[0] = 0

    _free = (ctypes.c_size_t * 1)()
    _free[0] = 0

    libcudart('cudaMemGetInfo', ctypes.byref(_free), ctypes.byref(_total))

    return int(_total[0]), int(_free[0])


#####################################################################################
# cuda_malloc
#####################################################################################

def cuda_malloc(d_ptr=None, num=None, dtype=None):
    """
    Allocate memory on device.
    :arg ctypes.ctypes_data d_ptr: Device pointer.
    :arg ctypes.c_int num: Number of elements.
    :arg ctypes.dtype dtype: Data type.
    """
    # TODO: make return error code.

    assert d_ptr is not None, "cuda_runtime:cuda_malloc error: no device pointer."
    assert num is not None, "cuda_runtime:cuda_malloc error: no length."
    assert dtype is not None, "cuda_runtime:cuda_malloc error: no type."


    libcudart('cudaMalloc', ctypes.byref(d_ptr), ctypes.c_size_t(num * ctypes.sizeof(dtype)))


#####################################################################################
# cuda_free
#####################################################################################

def cuda_free(d_ptr=None):
    """
    Free memory on device.
    :arg ctypes.ctypes_data d_ptr: Device pointer.
    """
    # TODO: make return error code.

    assert d_ptr is not None, "cuda_runtime:cuda_malloc error: no device pointer."

    libcudart('cudaFree', d_ptr)


#####################################################################################
# cuda_mem_cpy
#####################################################################################

def cuda_mem_cpy(d_ptr=None, s_ptr=None, size=None, cpy_type=None):
    """
    Copy memory between pointers.
    :arg ctypes.POINTER d_ptr: Destination pointer.
    :arg ctypes.POINTER s_ptr: Source pointer.
    :arg ctypes.c_size_t size: Number of bytes to copy.
    :arg str cpy_type: Type of copy.
    """
    # TODO: make return error code.

    assert d_ptr is not None, "cuda_runtime:cuda_mem_cpy error: no destination pointer."
    assert cpy_type is not None, "cuda_runtime:cuda_mem_cpy error: No copy type."
    assert s_ptr is not None, "cuda_runtime:cuda_mem_cpy error: no source pointer"
    assert type(size) is ctypes.c_size_t, "cuda_runtime:cuda_mem_cpy error: No size or size of incorrect type."

    assert cpy_type in ['cudaMemcpyHostToDevice', 'cudaMemcpyDeviceToHost', 'cudaMemcpyDeviceToDevice'], "cuda_runtime:cuda_mem_cpy error: No copy of that type."


    if cpy_type == 'cudaMemcpyHostToDevice':
        LIBHELPER['cudaCpyHostToDevice'](d_ptr, s_ptr, size)

    elif cpy_type == 'cudaMemcpyDeviceToHost':
        LIBHELPER['cudaCpyDeviceToHost'](d_ptr, s_ptr, size)

    elif cpy_type == 'cudaMemcpyDeviceToDevice':
        LIBHELPER['cudaCpyDeviceToDevice'](d_ptr, s_ptr, size)

    else:
        print "cuda_mem_cpy error: Something failed.", cpy_type, d_ptr, s_ptr, size





