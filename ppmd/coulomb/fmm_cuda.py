from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


from math import log, ceil
from ppmd.coulomb.octal import *
import numpy as np
from ppmd import runtime, host, kernel, pairloop, data, access, mpi, opt
from ppmd.lib import build
import ctypes
import os
import math
import cmath
from threading import Thread, Lock
from scipy.special import lpmv, rgamma, gammaincc, lambertw

from ppmd.cuda import *

REAL = ctypes.c_double
UINT64 = ctypes.c_uint64
UINT32 = ctypes.c_uint32
INT64 = ctypes.c_int64
INT32 = ctypes.c_int32

def red(input):
    try:
        from termcolor import colored
        return colored(input, 'red')
    except Exception as e: return input
def green(input):
    try:
        from termcolor import colored
        return colored(input, 'green')
    except Exception as e: return input
def yellow(input):
    try:
        from termcolor import colored
        return colored(input, 'yellow')
    except Exception as e: return input

def _numpy_ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)

def _check_dtype(arr, dtype):
    if arr.dtype != dtype:
        raise RuntimeError('Bad data type. Expected: {} Found: {}.'.format(
            str(dtype), str(arr.dtype)))
    if issubclass(type(arr), np.ndarray): return _numpy_ptr(arr)
    elif issubclass(type(arr), host.Matrix): return arr.ctypes_data
    elif issubclass(type(arr), host.Array): return arr.ctypes_data
    elif issubclass(type(arr), cuda_base.gpuarray.GPUArray):
        return ctypes.cast(arr.ptr, ctypes.c_void_p)
    else: raise RuntimeError('unknown array type passed: {}'.format(type(arr)))


_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

np.set_printoptions(threshold=np.nan)

class TranslateMTLCuda(object):
    def __init__(self, dtype, tree, nlevel, a_arr, ar_arr, p_arr, e_arr,
                 int_list, int_tlookup, int_plookup, int_radius, ipower_mtl,
                 wigner_f, wigner_b):
        self.tree = tree
        self.L = nlevel
        ncomp = (self.L**2) * 2
        self.tree_plain = OctalCudaDataTree(tree=tree, mode='plain',
                                            dtype=dtype, ncomp=ncomp)
        self.tree_halo = OctalCudaDataTree(tree=tree, mode='halo',
                                           dtype=dtype, ncomp=ncomp)



        self._d_a = cuda_base.gpuarray.to_gpu(a_arr)
        self._d_ar = cuda_base.gpuarray.to_gpu(ar_arr)
        self._d_p = cuda_base.gpuarray.to_gpu(p_arr)
        self._d_e = cuda_base.gpuarray.to_gpu(e_arr)

        self._int_list = []
        for lx in int_list:
            if lx is not None:
                ne = cuda_base.gpuarray.to_gpu(lx)
            else:
                ne = None
            self._int_list.append(ne)

        self._d_int_tlookup = cuda_base.gpuarray.to_gpu(int_tlookup)
        self._d_int_plookup = cuda_base.gpuarray.to_gpu(int_plookup)
        self._d_int_radius = cuda_base.gpuarray.to_gpu(int_radius)

        self._ipower_mtl = cuda_base.gpuarray.to_gpu(ipower_mtl)

        jlookup = np.zeros(ncomp, dtype=INT32)
        klookup = np.zeros(ncomp, dtype=INT32)

        ind = 0
        for jx in range(nlevel):
            for kx in range(-1*jx, jx+1):
                jlookup[ind] = jx
                klookup[ind] = kx
                ind += 1

        self._jlookup = cuda_base.gpuarray.to_gpu(jlookup)
        self._klookup = cuda_base.gpuarray.to_gpu(klookup)




        # need tmp space to rotate moments
        self.tmp_tree1 = OctalCudaDataTree(tree=tree, mode='plain',
                                           dtype=dtype, ncomp=ncomp)
        self.tmp_tree2 = OctalCudaDataTree(tree=tree, mode='plain',
                                           dtype=dtype, ncomp=ncomp)

        self._wigner_real = np.zeros((7,7,7), dtype=ctypes.c_void_p)
        self._wigner_imag = np.zeros((7,7,7), dtype=ctypes.c_void_p)

        self._wigner_b_real = np.zeros((7,7,7), dtype=ctypes.c_void_p)
        self._wigner_b_imag = np.zeros((7,7,7), dtype=ctypes.c_void_p)

        self._dev_matrices = []
        self._dev_pointers = []

        # convert host rotation matrices to device matrices
        for iz, pz in enumerate(range(-3, 4)):
            for iy, py in enumerate(range(-3, 4)):
                for ix, px in enumerate(range(-3, 4)):

                    # forward real
                    f = wigner_f[(pz, py, px)]

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):
                        # forward real
                        nn = cuda_base.gpuarray.to_gpu(f['real'][p])
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # forward real
                    self._wigner_real[pz, py, px] = self._dev_pointers[-1].ptr

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):
                        # forward imag
                        nn = cuda_base.gpuarray.to_gpu(f['imag'][p])
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # forward imag
                    self._wigner_imag[pz, py, px] = self._dev_pointers[-1].ptr


                    f = None
                    # backward real
                    b = wigner_b[(pz, py, px)]

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):
                        # backward real
                        nn = cuda_base.gpuarray.to_gpu(b['real'][p])
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # backward real
                    self._wigner_b_real[pz, py, px] =self._dev_pointers[-1].ptr

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):
                        # backward imag
                        nn = cuda_base.gpuarray.to_gpu(b['imag'][p])
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # backward imag
                    self._wigner_b_imag[pz, py, px] =self._dev_pointers[-1].ptr


        # pointers to pointers on device
        self._wigner_real   = cuda_base.gpuarray.to_gpu(self._wigner_real)
        self._wigner_imag   = cuda_base.gpuarray.to_gpu(self._wigner_imag)
        self._wigner_b_real = cuda_base.gpuarray.to_gpu(self._wigner_b_real)
        self._wigner_b_imag = cuda_base.gpuarray.to_gpu(self._wigner_b_imag)




        # load multipole to local lib
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaTranslateMTLZ.cu') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaTranslateMTLZ.h') as fh:
            hpp = fh.read()
        self._translate_mtl_lib = cuda_build.simple_lib_creator(hpp, cpp,
            'fmm_translate_mtl')

        self.timer_mtl = opt.Timer(runtime.TIMER)

        self._lock = Lock()


    def translate_mtl_pre(self, level, host_halo_tree):
        self.tree_halo[level] = host_halo_tree

    def translate_mtl_async_func(self, level, radius):
        self._translate_mtl(level, radius)

    def translate_mtl_post(self, level, host_plain_tree=None):
        if host_plain_tree is not None:
            self.tree_plain.get(level, host_plain_tree)
            return None

        return self.tree_plain[level]


    # perform all mtl stages in one step
    def translate_mtl(self, host_halo_tree, level, radius,
                      host_plain_tree=None):

        self.tree_halo[level] = host_halo_tree
        self._translate_mtl(level, radius)
        if host_plain_tree is not None:
            self.tree_plain.get(level, host_plain_tree)
            return None
        return self.tree_plain[level]

    def _translate_mtl(self, level, radius):
        self._lock.acquire(True)

        self.timer_mtl.start()
        print("DEVICE_NUMBER:\t",cuda_runtime.DEVICE_NUMBER)
        err = self._translate_mtl_lib['translate_mtl'](
            _check_dtype(self.tree[level].local_grid_cube_size, UINT32),
            self.tree_halo.device_pointer(level),
            self.tree_plain.device_pointer(level),
            _check_dtype(self._d_e, REAL),
            _check_dtype(self._d_p, REAL),
            _check_dtype(self._d_a, REAL),
            _check_dtype(self._d_ar, REAL),
            REAL(radius),
            INT64(self.L),
            _check_dtype(self._int_list[level], INT32),
            _check_dtype(self._d_int_tlookup, INT32),
            _check_dtype(self._d_int_plookup, INT32),
            _check_dtype(self._d_int_radius, ctypes.c_double),
            _check_dtype(self._jlookup, INT32),
            _check_dtype(self._klookup, INT32),
            _check_dtype(self._ipower_mtl, REAL),
            INT32(128),
            INT32(cuda_runtime.DEVICE_NUMBER)
        )

        self.timer_mtl.pause()
        self._lock.release()

        cuda_runtime.cuda_err_check(err)

        if err < 0:
            raise RuntimeError("Negative error code caught: {}".format(err))
































