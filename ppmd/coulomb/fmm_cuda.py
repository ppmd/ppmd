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
from threading import Thread
from scipy.special import lpmv, rgamma, gammaincc, lambertw

from ppmd.cuda import *

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


_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

np.set_printoptions(threshold=np.nan)

class TranslateMTLCuda(object):
    def __init__(self, dtype, tree, nlevel, a_arr, ar_arr, p_arr, e_arr,
                 int_tlookup, int_plookup, int_radius):
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

        self._d_int_tlookup = cuda_base.gpuarray.to_gpu(int_tlookup)
        self._d_int_plookup = cuda_base.gpuarray.to_gpu(int_plookup)
        self._d_int_radius = cuda_base.gpuarray.to_gpu(int_radius)

        # load multipole to local lib
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaTranslateMTL.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaTranslateMTL.h') as fh:
            hpp = fh.read()
        self._translate_mtl_lib = cuda_build.simple_lib_creator(hpp, cpp,
            'fmm_translate_mtl')



















