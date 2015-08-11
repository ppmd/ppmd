"""
Module to hold CUDA related code.
"""
import os
import ctypes as ct
import data
import build
import subprocess
import hashlib
import atexit

ERROR_LEVEL = build.Level(1)


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

def _build_static_libs(lib):

    with open("./lib/" + lib + ".cu", "r") as fh:
        _code = fh.read()
        fh.close()
    with open("./lib/" + lib + ".h", "r") as fh:
        _code += fh.read()
        fh.close()

    _m = hashlib.md5()
    _m.update(_code)
    _m = _m.hexdigest()

    _lib_filename = os.path.join('./build/', lib + '_' +str(_m) +'.so')

    if data.MPI_HANDLE.rank == 0:
        if not os.path.exists(_lib_filename):

            _lib_src_filename = './lib/' + lib + '.cu'
            _c_cmd = NVCC.binary + [_lib_src_filename] + ['-o'] + [_lib_filename] + NVCC.l_flags
            if build.DEBUG.level > 0:
                _c_cmd += NVCC.dbg_flags
            _c_cmd += NVCC.shared_lib_flag

            if build.VERBOSE.level > 1:
                data.pprint("gpucuda _build_static_libs compile cmd:", _c_cmd)

            stdout_filename = './build/' + lib + '_' +str(_m) + '.log'
            stderr_filename = './build/' + lib + '_' +str(_m) + '.err'
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
                if ERROR_LEVEL.level > 2:
                    raise RuntimeError('gpucuda error: helper library not built.')
                elif build.VERBOSE.level > 2:
                    data.pprint("gpucuda warning: Shared library not built:", lib)


    data.MPI_HANDLE.barrier()

    return _lib_filename


#####################################################################################
# load libraries
#####################################################################################

try:
    CUDA_INC_PATH = os.environ['CUDA_INC_PATH']
except KeyError:
    if ERROR_LEVEL.level > 2:
        raise RuntimeError('gpucuda error: cuda toolkit environment path not found, expecting CUDA_INC_PATH')
    CUDA_INC_PATH = None

try:
    LIBCUDART = ct.cdll.LoadLibrary(CUDA_INC_PATH + "/lib64/libcudart.so.6.5")
except:
    if ERROR_LEVEL.level > 2:
        raise RuntimeError('gpucuda error: Module is not initialised correctly, CUDA runtime not loaded')
    LIBCUDART = None

# wrapper library for functions involving types.
try:
    LIBHELPER = ct.cdll.LoadLibrary(_build_static_libs('cudaHelperLib'))
except:
    LIBHELPER = None


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
        DEVICE.id = _r
    else:
        data.rprint("gpucuda warning: No device set")

    if (build.VERBOSE.level > 0) and (LIBCUDART is not None):
        _dev = ct.c_int()
        LIBCUDART['cudaGetDevice'](ct.byref(_dev))
        data.rprint("cudaGetDevice returned device ", _dev.value)

#####################################################################################
# Is module ready to use?
#####################################################################################

def INIT_STATUS():
    """
    Function to determine if the module is correctly loaded and can be used.
    :return: True/False.
    """

    if (LIBCUDART is not None) and (LIBHELPER is not None) and (DEVICE.id is not None):
        return True
    else:
        return False


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
        data.rprint(args)

    cuda_err_check(LIBCUDART[args[0]](*args[1::]))

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

        # create device pointer.
        self._d_p = ct.POINTER(self._dtype)()

        #allocate space.
        libcudart('cudaMalloc', ct.byref(self._d_p), ct.c_size_t(self._size * ct.sizeof(self._dtype)))

        # Bool to flag if memory is allocated on device.
        self._alloc = True


    def free(self):
        """
        Free allocated device memory. Do not call after a device reset.
        :return:
        """
        libcudart('cudaFree', self._d_p)


    def cpy_htd(self, host_ptr, size=None):
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

    def cpy_dth(self, host_ptr, size=None):
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

    @property
    def dtype(self):
        """
        :return: Data type of array.
        """
        return self._dtype

    @property
    def ctypes_data(self):
        """
        Get pointer to device memory.
        :return: device pointer
        """
        return self._d_p



#####################################################################################
# Module Init
#####################################################################################

cuda_set_device()

#####################################################################################
# Module cleanup
#####################################################################################

def gpucuda_cleanup():
    cuda_device_reset()

atexit.register(gpucuda_cleanup)
