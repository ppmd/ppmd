from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


from ppmd import runtime, host, pairloop, data, mpi, opt
from ppmd.lib import build
from ppmd.access import *

import ctypes
import os
import numpy as np

_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

REAL = ctypes.c_double
INT64 = ctypes.c_int64
INT32 = ctypes.c_int32

class FMMLocal(object):
    """
    Class to perform local part of fmm
    """
    def __init__(self, width, domain, entry_data, entry_map, free_space,
            dtype, force_unit, energy_unit):

        self.width = width
        self.domain = domain
        self.entry_data = entry_data
        self.entry_map = entry_map
        self.free_space = free_space
        self.dtype = dtype

        self.sh = pairloop.state_handler.StateHandler(state=None, shell_cutoff=width)

        with open(str(_SRC_DIR) + \
                          '/FMMSource/LocalCells.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/LocalCells.h') as fh:
            hpp = fh.read()

        hpp = hpp % {
            'SUB_FORCE_UNIT': str(force_unit),
            'SUB_ENERGY_UNIT': str(energy_unit)
        }

        self._lib = build.simple_lib_creator(hpp, cpp, 'fmm_local')['local_cell_by_cell']

        self._global_size = np.zeros(3, dtype=INT64)
        self._global_size[:] = entry_map.cube_side_count

        self._ncells =  (self._global_size[0] + 6) * \
                        (self._global_size[1] + 6) * \
                        (self._global_size[2] + 6)

        self._local_size = np.zeros(3, dtype=INT64)
        self._local_size[:] = self.entry_data.local_size[:]
        
        self._local_offset = np.zeros(3, dtype=INT64)
        self._local_offset[:] = self.entry_data.local_offset[:]
        self._u = np.zeros(1, dtype=self.dtype)

        self._ll_array = np.zeros(1, dtype=INT64)
        self._ll_ccc_array = np.zeros(self._ncells, dtype=INT64)

        bn = 10
        self._tmp_n = bn
        self._tmp_int_i = host.ThreadSpace(n=bn, dtype=INT64)
        self._tmp_int_j = host.ThreadSpace(n=bn, dtype=INT64)
        self._tmp_real_pi = host.ThreadSpace(n=bn, dtype=REAL)
        self._tmp_real_pj = host.ThreadSpace(n=bn, dtype=REAL)
        self._tmp_real_qi = host.ThreadSpace(n=bn, dtype=REAL)
        self._tmp_real_qj = host.ThreadSpace(n=bn, dtype=REAL)
        self._tmp_real_fi = host.ThreadSpace(n=bn, dtype=REAL)

    def __call__(self, positions, charges, forces, cells):
        """
        const INT64 free_space,
        const INT64 * RESTRICT global_size,
        const INT64 * RESTRICT local_size,
        const INT64 * RESTRICT local_offset,
        const INT64 num_threads,
        const INT64 nlocal,
        const INT64 ntotal,
        const REAL * RESTRICT P,
        const REAL * RESTRICT Q,
        const REAL * RESTRICT C,
        REAL * RESTRICT F,
        REAL * RESTRICT U,
        INT64 * RESTRICT ll_array,
        INT64 * RESTRICT ll_ccc_array,
        INT64 * RESTRICT * RESTRICT tmp_int_i,
        INT64 * RESTRICT * RESTRICT tmp_int_j,
        REAL * RESTRICT * RESTRICT tmp_real_pi,
        REAL * RESTRICT * RESTRICT tmp_real_pj,
        REAL * RESTRICT * RESTRICT tmp_real_qi,
        REAL * RESTRICT * RESTRICT tmp_real_qj,
        REAL * RESTRICT * RESTRICT tmp_real_fi        
        """
        dats = {
            'p': positions(READ),
            'q': charges(READ),
            'f': forces(INC),
            'c': cells(READ)
        }
        
        self._u[0] = 0.0

        nlocal, nhalo, ncell = self.sh.pre_execute(dats=dats)
        ntotal = nlocal + nhalo
        
        if self._ll_array.shape[0] < (ntotal + self._ncells):
            self._ll_array = np.zeros(ntotal+100+self._ncells, dtype=INT64)

        if self._tmp_n < ncell*15:
            bn = ncell*15 + 100
            self._tmp_int_i = host.ThreadSpace(n=bn, dtype=INT64)
            self._tmp_int_j = host.ThreadSpace(n=bn, dtype=INT64)
            self._tmp_real_pi = host.ThreadSpace(n=3*bn, dtype=REAL)
            self._tmp_real_pj = host.ThreadSpace(n=3*bn, dtype=REAL)
            self._tmp_real_qi = host.ThreadSpace(n=bn, dtype=REAL)
            self._tmp_real_qj = host.ThreadSpace(n=bn, dtype=REAL)
            self._tmp_real_fi = host.ThreadSpace(n=3*bn, dtype=REAL)
            self._tmp_n = bn
        
        #print("\ttmp_n", self._tmp_n, "nlocal", nlocal, "nhalo", nhalo, "max_cell", ncell)

        #for px in range(ntotal):
        #    print(px, cells[px], "\t", positions[px,:], charges[px,:])
        
        if self.domain.extent.dtype is not REAL:
            raise RuntimeError("expected c_double extent")
        
        if self.free_space == '27':
            free_space = 0
        elif self.free_space == True:
            free_space = 1
        else:
            free_space = 0

        err = self._lib(
            INT64(free_space),
            self.domain.extent.ctypes_data,
            self._global_size.ctypes.get_as_parameter(),
            self._local_size.ctypes.get_as_parameter(),
            self._local_offset.ctypes.get_as_parameter(),
            INT64(runtime.NUM_THREADS),
            INT64(nlocal),
            INT64(ntotal),
            self.sh.get_pointer(positions(READ)),
            self.sh.get_pointer(charges(READ)),
            self.sh.get_pointer(cells(READ)),
            self.sh.get_pointer(forces(INC)),
            self._u.ctypes.get_as_parameter(),
            self._ll_array.ctypes.get_as_parameter(),
            self._ll_ccc_array.ctypes.get_as_parameter(),
            self._tmp_int_i.ctypes_data,
            self._tmp_int_j.ctypes_data,
            self._tmp_real_pi.ctypes_data,
            self._tmp_real_pj.ctypes_data,
            self._tmp_real_qi.ctypes_data,
            self._tmp_real_qj.ctypes_data,
            self._tmp_real_fi.ctypes_data
        )

        self.sh.post_execute(dats=dats)
        if err < 0:
            raise RuntimeError("Negative error code: {}".format(err))
    
        return self._u[0]











