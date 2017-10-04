__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import scipy
from math import log, ceil
from ppmd.coulomb.octal import *
import numpy as np
from ppmd import runtime
import ctypes

class PyFMM(object):
    def __init__(self, domain, N, eps=10.**-6, shared_memory=False,
                 dtype=ctypes.c_double):

        self.L = int(-1*log(eps,2))
        """Number of multipole expansion coefficients"""
        self.R = int(ceil(log(N, 8)))
        """Number of levels in octal tree."""
        self.dtype = dtype
        """Floating point datatype used."""

        # define the octal tree and attach data to the tree.
        self.tree = OctalTree(self.R, domain.comm)
        self.tree_plain = OctalDataTree(self.tree, self.L, 'plain', dtype)
        self.tree_halo = OctalDataTree(self.tree, self.L, 'halo', dtype)
        self.tree_parent = OctalDataTree(self.tree, self.L, 'parent', dtype)

        self._tcount = runtime.OMP_NUM_THREADS if runtime.OMP_NUM_THREADS is \
            not None else 1
        tmp_size = self.L
        self._thread_tmp = [np.zeros(tmp_size, dtype=dtype) for tx
                            in range(self._tcount)]
        self._thread_tmp_pointers = (ctypes.POINTER(dtype) * self._tcount)(
            *[kx.ctypes.data_as(ctypes.POINTER(dtype)) for
              kx in self._thread_tmp]
        )

    def _compute_cube_contrib(self, positions):
        pass










