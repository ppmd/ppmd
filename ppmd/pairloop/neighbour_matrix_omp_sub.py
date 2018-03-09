# system level
from __future__ import division, print_function

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import numpy as np
import ctypes, os

from ppmd import runtime, opt
from ppmd.lib import build

INT64 = ctypes.c_int64
REAL = ctypes.c_double

################################################################################################################
# NeighbourList definition 27 cell version
################################################################################################################

class NeighbourListOMPSub(object):
    def __init__(self, cell_width, cell_list, n=100):

        self.cell_width = cell_width
        self.cell_list = cell_list    
        
        if self.cell_list.cell_width < cell_width:
            raise RuntimeError('cell list has smaller cells than requested')

        self.timer_update = ppmd.opt.Timer(runtime.TIMER)

        self.matrix = np.zeros(1, dtype=INT64)
        self.ncount = np.zeros(n, dtype=INT64)
        self.stride = INT64(0)
        self.n_local = 0
        self.total_num_neighbours = 0
        self.max_size = 0

        bn = os.path.join(os.path.dirname(__file__), 'lib')
        bn += '/NeighbourMatrixSourceSub'
        self._lib = build.lib_from_file_source(bn, 'OMPNeighbourMatrixSub')[
            'OMPNeighbourMatrixSub'
        ]


    def update(self, npart_local, positions):
        if positions.dtype is not REAL:
            raise RuntimeError('positions must have dtype ctypes.c_double')

        self.timer_update.start()
        
        n = npart_local
        needed_stride = self.cell_list.max_cell_contents_count*27

        if self.stride.value < needed_stride:
            self.stride.value = needed_stride
        if self.ncount.shape[0] < n:
            self.ncount = np.zeros(n+100, dtype=INT64)
        if self.matrix.shape[0] < n*needed_stride:
            self.matrix = np.zeros((n+100)*needed_stride, dtype=INT64)

        _nt = INT64(0)
        ret = self._lib(
            INT64(npart_local),
            positions.ctypes_data,
            self.cell_list.list.ctypes.get_as_parameter(),
            self.cell_list.cell_offset,
            self.cell_list.cell_reverse_lookup.ctypes.get_as_parameter(),
            self.cell_list.cell_array.ctypes.get_as_parameter(),
            self.matrix.ctypes.get_as_parameter(),
            self.ncount.ctypes.get_as_parameter(),
            self.stride,
            REAL(self.cell_width),
            ctypes.byref(_nt)
        )
        assert ret >= 0, "lib failed, return code: " + str(ret)
        
        self.total_num_neighbours = _nt.value
        self.n_local = npart_local

        self.timer_update.pause()
        opt.PROFILE[
            self.__class__.__name__+':update('+str(self.cell_width)+')'
        ] = (self.timer_update.time())

        self.max_size = max(self.max_size, self.matrix.size)

        opt.PROFILE[
            self.__class__.__name__+':nbytes('+str(self.cell_width)+')'
        ] = (self.max_size)



