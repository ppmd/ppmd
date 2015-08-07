"""
Module to hold CUDA related code.
"""
import os
import ctypes
import data
import build

try:
    CUDA_INC_PATH = os.environ['CUDA_INC_PATH']
except KeyError:
    CUDA_INC_PATH = None

try:
    LIBCUDART = ctypes.cdll.LoadLibrary(CUDA_INC_PATH + "/lib64/libcudart.so")
except:
    LIBCUDART = None

def cuda_set_device(device=None):

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

        LIBCUDART['cudaSetDevice'](ctypes.c_int(_r))
    else:
        data.rprint("gpucuda warning: No device set")


    if (build.VERBOSE.level > 0) and (LIBCUDART is not None):
        dev = ctypes.c_int()
        LIBCUDART['cudaGetDevice'](ctypes.byref(dev))
        data.rprint("cudaGetDevice returned device ", dev.value)




