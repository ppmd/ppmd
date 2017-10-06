__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import scipy
from math import log, ceil
from ppmd.coulomb.octal import *
import numpy as np
from ppmd import runtime
from ppmd.lib import build
import ctypes
import os


_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

REAL = ctypes.c_double
UINT64 = ctypes.c_uint64
INT32 = ctypes.c_int32

def _numpy_ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)

def _check_dtype(arr, dtype):
    if arr.dtype != dtype:
        raise RuntimeError('Bad data type. Expected: {} Found: {}.'.format(
            str(dtype), str(arr.dtype)))

class PyFMM(object):
    def __init__(self, domain, N, eps=10.**-6, shared_memory=False,
                 dtype=ctypes.c_double):

        self.L = int(-1*log(eps,2))
        """Number of multipole expansion coefficients"""
        self.R = int(ceil(log(N, 8)))
        """Number of levels in octal tree."""
        self.dtype = dtype
        """Floating point datatype used."""
        self.domain = domain

        # define the octal tree and attach data to the tree.
        self.tree = OctalTree(self.R, domain.comm)
        self.tree_plain = OctalDataTree(self.tree, self.L, 'plain', dtype)
        self.tree_halo = OctalDataTree(self.tree, self.L, 'halo', dtype)
        self.tree_parent = OctalDataTree(self.tree, self.L, 'parent', dtype)
        self.entry_data = EntryData(self.tree, self.L, ctypes.c_double)

        self._tcount = runtime.OMP_NUM_THREADS if runtime.OMP_NUM_THREADS is \
            not None else 1
        self._thread_allocation = np.zeros(1, dtype=INT32)

        # load contribution conputation
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleContribution.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleContribution.h') as fh:
            hpp = fh.read()
        self._contribution_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_contrib')['particle_contribution']






    def _compute_cube_contrib(self, positions):


        '''
        const UINT64 npart,
        const INT32 thread_max,
        const REAL * RESTRICT position,             // xyz
        const REAL * RESTRICT boundary,             // xl. xu, yl, yu, zl, zu
        const UINT64 * RESTRICT cube_offset,        // zyx (slowest to fastest)
        const UINT64 * RESTRICT cube_dim,           // as above
        const UINT64 * RESTRICT cube_side_counts,   // as above
        REAL * RESTRICT cube_data,                  // lexicographic
        INT32 * RESTRICT thread_assign
        '''

        ns = self.tree.entry_map.cube_side_count
        cube_side_counts = np.array((ns, ns, ns), dtype=UINT64)
        if self._thread_allocation.size < self._tcount*positions.npart_local:
            self._thread_allocation = np.zeros(
                self._tcount*positions.npart_local,dtype=INT32)
        else:
            self._thread_allocation[:self._tcount:] = 0

        _check_dtype(positions, REAL)
        _check_dtype(self.domain.extent_internal, REAL)
        _check_dtype(self.entry_data.local_offset, UINT64)
        _check_dtype(self.entry_data.local_size, UINT64)
        _check_dtype(cube_side_counts, UINT64)
        _check_dtype(self.entry_data.data, REAL)
        _check_dtype(self._thread_allocation, INT32)

        err = self._contribution_lib(
            UINT64(positions.npart_local),
            INT32(self._tcount),
            positions.ctypes_data,
            self.domain.extent_internal.ctypes_data,
            _numpy_ptr(self.entry_data.local_offset),
            _numpy_ptr(self.entry_data.local_size),
            _numpy_ptr(cube_side_counts),
            _numpy_ptr(self.entry_data.data),
            _numpy_ptr(self._thread_allocation)
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))
















