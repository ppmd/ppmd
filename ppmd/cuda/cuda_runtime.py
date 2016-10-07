"""
Module to handle the cuda runtime environment.
"""
#system level imports
import ctypes
import os
import math
import atexit

# pycuda imports
import pycuda.driver as cudadrv

#package level imports
import cuda_config
from ppmd import runtime, pio, mpi

CUDA_ENABLED = cuda_config.CUDA_CFG['enable-cuda'][1]


OPT = cuda_config.CUDA_CFG['opt-level'][1]
DEBUG = cuda_config.CUDA_CFG['debug-level'][1]
VERBOSE = cuda_config.CUDA_CFG['verbose-level'][1]
TIMER = cuda_config.CUDA_CFG['timer-level'][1]
BUILD_TIMER = cuda_config.CUDA_CFG['build-timer-level'][1]
ERROR_LEVEL = cuda_config.CUDA_CFG['error-level'][1]


BUILD_DIR = runtime.BUILD_DIR

LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/'))

# Init cuda
cudadrv.init()

import cuda_build

try:
    LIB_HELPER = ctypes.cdll.LoadLibrary(cuda_build.build_static_libs('cudaHelperLib'))
except Exception as e:
    print e
    raise RuntimeError('cuda_runtime error: Module is not initialised correctly, CUDA helper lib not loaded')



###############################################################################
# cuda_err_checking
###############################################################################

def cuda_err_check(err_code):
    """
    Wrapper to check cuda error codes.
    :param err_code:
    :return:
    """

    assert LIB_HELPER is not None, "cuda_runtime error: No error checking library"

    if LIB_HELPER is not None:
        err = LIB_HELPER['cudaErrorCheck'](err_code)
        assert err == 0, "Non-zero CUDA error:" + str(err_code)

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


        device_count = ctypes.c_int()
        device_count.value = 0
        cuda_err_check(LIB_HELPER['cudaGetDeviceCountWrapper'](ctypes.byref(device_count)))
        assert device_count != 0, "CUDA Device count query returned zero!"

        if _mv2r is not None:
            _r = int(_mv2r) % device_count.value
        elif _ompr is not None:
            _r = int(_ompr) % device_count.value
        else:
            _r = mpi.MPI_HANDLE.nproc % device_count.value

        print _r, device_count.value, mpi.MPI_HANDLE.rank, mpi.MPI_HANDLE.nproc
        return cudadrv.Device(_r)

    else:
        return cudadrv.Device(device)

# Set device
DEVICE = cuda_set_device()
# Make context
CONTEXT = DEVICE.make_context()
# Register destruction
atexit.register(CONTEXT.pop)



try:
    CUDA_INC_PATH = os.environ['CUDA_INSTALL_PATH']
except KeyError:
    if ERROR_LEVEL > 2:
        raise RuntimeError('cuda_runtime error: cuda toolkit environment path not found, expecting CUDA_INSTALL_PATH')
    CUDA_INC_PATH = None

try:
    LIB_CUDART = ctypes.cdll.LoadLibrary(CUDA_INC_PATH + "/lib64/libcudart.so")

except:
    if ERROR_LEVEL > 2:
        raise RuntimeError('cuda_runtime error: Module is not initialised correctly, CUDA runtime not loaded')
    LIB_CUDART = None

try:
    LIB_CUDA_MISC = ctypes.cdll.LoadLibrary(cuda_build.build_static_libs('cudaMisc'))
except:
    raise RuntimeError('cuda_runtime error: Module is not initialised correctly, CUDA Misc lib not loaded')








###############################################################################
# CUDA runtime handle
###############################################################################

def libcudart(*args):
    """
    Wrapper to cuda runtime library with error code checking.
    :param args: string <function name>, args.
    :return:
    """

    assert LIB_CUDART is not None, "cuda_runtime error: No CUDA Runtime library loaded"

    if VERBOSE > 2:
        pio.rprint(args)

    cuda_err_check(LIB_CUDART[args[0]](*args[1::]))



###############################################################################
# Is module ready to use?
###############################################################################


def INIT_STATUS():
    """
    Function to determine if the module is correctly loaded and can be used.
    :return: True/False.
    """

    if (LIB_CUDART is not None) and\
            (LIB_HELPER is not None) and\
            (DEVICE.id is not None) and\
            CUDA_ENABLED:
        return True
    else:
        return False



###############################################################################
#  cuMemGetInfo
###############################################################################

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


###############################################################################
# cuda_malloc
###############################################################################

def cuda_malloc(d_ptr=None, num=None, dtype=None):
    """
    Allocate memory on device.
    :arg ctypes.ctypes_data d_ptr: Device pointer.
    :arg int num: Number of elements.
    :arg ctypes.dtype dtype: Data type.
    """
    # TODO: make return error code.

    assert d_ptr is not None, "cuda_runtime:cuda_malloc error: no device pointer."
    assert num is not None, "cuda_runtime:cuda_malloc error: no length."
    assert dtype is not None, "cuda_runtime:cuda_malloc error: no type."


    libcudart('cudaMalloc', ctypes.byref(d_ptr), ctypes.c_size_t(num * ctypes.sizeof(dtype)))


###############################################################################
# cuda_free
###############################################################################

def cuda_free(d_ptr=None):
    """
    Free memory on device.
    :arg ctypes.ctypes_data d_ptr: Device pointer.
    """
    # TODO: make return error code.

    assert d_ptr is not None, "cuda_runtime:cuda_malloc error: no device pointer."

    libcudart('cudaFree', d_ptr)


###############################################################################
# cuda_mem_cpy
###############################################################################

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
        cuda_err_check(LIB_HELPER['cudaCpyHostToDevice'](d_ptr, s_ptr, size))

    elif cpy_type == 'cudaMemcpyDeviceToHost':
        cuda_err_check(LIB_HELPER['cudaCpyDeviceToHost'](d_ptr, s_ptr, size))

    elif cpy_type == 'cudaMemcpyDeviceToDevice':
        cuda_err_check(LIB_HELPER['cudaCpyDeviceToDevice'](d_ptr, s_ptr, size))

    else:
        print "cuda_mem_cpy error: Something failed.", cpy_type, d_ptr, s_ptr, size


###############################################################################
# Make cuda 1D threadblock
###############################################################################
def kernel_launch_args_1d(n=None, threads=512):
    """
    Given a n return cuda launch args for a kernel requiring at least n threads.
    """
    assert n is not None, "No target number of threads passed"

    _blocksize = (ctypes.c_int * 3)(int(math.ceil(n / float(threads))), 1, 1)
    _threadsize = (ctypes.c_int * 3)(threads, 1, 1)


    return _blocksize, _threadsize



def cuda_exclusive_scan(array, length):
    if type(array.dtype) is ctypes.c_int:
        LIB_CUDA_MISC['cudaExclusiveScanInt'](array.ctypes_data,
                                              ctypes.c_int(length))
    elif type(array.dtype) is ctypes.c_double:
        LIB_CUDA_MISC['cudaExclusiveScanDouble'](array.ctypes_data,
                                                 ctypes.c_int(length))
    else:
        print "Fatal error: Array type does not have a corresponding exclusive" \
              "scan. Type is:", array.dtype
        quit()


