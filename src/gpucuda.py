"""
Module to hold CUDA related code.
"""
import os
import ctypes as ct
import data
import build
import subprocess



#####################################################################################
# NVCC Compiler
#####################################################################################

NVCC = build.Compiler(['nvcc_system_default'],
                ['nvcc'],
                ['-O3'],
                ['-lm'],
                ['-g'],
                ['-c'],
                ['-shared', '-Xcompiler','-fPIC'])

#####################################################################################
# build static libs
#####################################################################################

def _build_static_libs():

    _libs = ['cudaHelperLib']
    for lib in _libs:

        if not os.path.exists(os.path.join('./lib/', lib + '.so')):

            _lib_filename = './lib/' + lib + '.so'
            _lib_src_filename = './lib/' + lib + '.cu'
            _c_cmd = NVCC.binary + [_lib_src_filename] + ['-o'] + [_lib_filename] + NVCC.l_flags
            if build.DEBUG.level > 0:
                _c_cmd += NVCC.dbg_flags
            _c_cmd += NVCC.shared_lib_flag

            if build.VERBOSE.level > 1:
                print "gpucuda _build_static_libs compile cmd:", _c_cmd

            stdout_filename = './lib/' + lib + '.log'
            stderr_filename = './lib/' + lib + '.err'
            try:
                with open(stdout_filename, 'w') as stdout:
                    with open(stderr_filename, 'w') as stderr:
                        stdout.write('Compilation command:\n')
                        stdout.write(' '.join(_c_cmd))
                        stdout.write('\n\n')
                        p = subprocess.Popen(_c_cmd,
                                             stdout=stdout,
                                             stderr=stderr)
                        p.communicate()
            except:
                print "gpucuda error: Shared library not built:", lib

        if not os.path.exists(os.path.join('./lib/', lib + '.so')):
            print "gpucuda error: Shared library not build:", lib



_build_static_libs()


#####################################################################################
# load cuda runtime library
#####################################################################################

try:
    CUDA_INC_PATH = os.environ['CUDA_INC_PATH']
except KeyError:
    CUDA_INC_PATH = None

try:
    LIBCUDART = ct.cdll.LoadLibrary(CUDA_INC_PATH + "/lib64/libcudart.so.6.5")
except:
    LIBCUDART = None

try:
    LIBHELPER = ct.cdll.LoadLibrary("./lib/cudaHelperLib.so")
except:
    LIBHELPER = None



#####################################################################################
# cuda_set_device
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
            print("gpucuda warning: Did not find local rank, defaulting to device 0")

        if (_mv2r is not None) and (_ompr is not None):
            print("gpucuda warning: Found two local ranks, defaulting to device 0")

        if _mv2r is not None:
            _r = int(_mv2r) % data.MPI_HANDLE.nproc
        if _ompr is not None:
            _r = int(_ompr) % data.MPI_HANDLE.nproc

    else:
        _r = int(device)

    if LIBCUDART is not None:
        if build.VERBOSE.level > 0:
            data.rprint("setting device ", _r)

        LIBCUDART['cudaSetDevice'](ct.c_int(_r))
    else:
        data.rprint("gpucuda warning: No device set")

    if (build.VERBOSE.level > 0) and (LIBCUDART is not None):
        _dev = ct.c_int()
        LIBCUDART['cudaGetDevice'](ct.byref(_dev))
        data.rprint("cudaGetDevice returned device ", _dev.value)


#####################################################################################
# cuda_err_checking
#####################################################################################

def cuda_err_check(err_code):
    """
    Wrapper to check cuda error codes.
    :param err_code:
    :return:
    """
    if LIBHELPER is not None:
        LIBHELPER['cudaErrorCheck'](err_code)


def libcudart(*args):
    """
    Wrapper to cuda runtime library with error code checking.
    :param args: string <function name>, args.
    :return:
    """

    assert LIBCUDART is not None, "gpucuda error: No CUDA Runtime library loaded"

    if build.VERBOSE.level > 2:
        print args

    cuda_err_check(LIBCUDART[args[0]](*args[1::]))

#####################################################################################
# cuda_device_reset
#####################################################################################


def cuda_device_reset():
    """
    Reset the current cuda device.
    :return:
    """
    libcudart('cudaDeviceReset')

#####################################################################################
# CudaDeviceDat
#####################################################################################


class CudaDeviceDat(object):
    """
    Class to handle memory on cuda devices.
    :arg int size: Number of elements in array.
    :arg dtype: Data type default ctypes.c_double.
    """

    def __init__(self, size=1, dtype=ct.c_double):

        self._size = size
        self._dtype = dtype

        #create device pointer.
        self._d_p = ct.POINTER(self._dtype)()

        #allocate space.
        libcudart('cudaMalloc', ct.byref(self._d_p), ct.c_size_t(self._size * ct.sizeof(self._dtype)))


    def free(self):
        """
        Free allocated device memory.
        :return:
        """
        libcudart('cudaFree', self._d_p)

    def cpy_host_to_device(self, host_ptr, size=None):
        """
        Copy data from host pointer to device.
        :param host_ptr: host pointer to copy from.
        :param size: amount to copy (bytes). If not specified Method will copy with size equal to the length
        of the allocated device array.
        :return:
        """


        if size is None:
            _s = ct.c_size_t(self._size * ct.sizeof(self._dtype))
        else:
            _s = ct.c_size_t(size)

        LIBHELPER['cudaCpyHostToDevice'](self._d_p, host_ptr,_s)

    def cpy_device_to_host(self, host_ptr, size=None):
        """
        Copy data from host pointer to device.
        :param host_ptr: host pointer to copy into.
        :param size: amount to copy (bytes). If not specified Method will copy with size equal to the length
        of the allocated device array.
        :return:
        """


        if size is None:
            _s = ct.c_size_t(self._size * ct.sizeof(self._dtype))
        else:
            _s = ct.c_size_t(size)

        LIBHELPER['cudaCpyDeviceToHost'](host_ptr, self._d_p, _s)




