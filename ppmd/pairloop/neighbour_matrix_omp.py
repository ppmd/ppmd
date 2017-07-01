# system level
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes
from ppmd import build, host, runtime, opt


################################################################################################################
# NeighbourList definition 27 cell version
################################################################################################################

class NeighbourListOMP(object):
    def __init__(self, domain, cell_width, cell_list):
        self._domain = domain
        self.cell_width = cell_width
        self.cell_list = cell_list
        self.version_id = 0
        self.timer_update = opt.Timer(runtime.TIMER)
        self.matrix = host.Array(ncomp=1, dtype=ctypes.c_int)
        self.ncount = host.Array(ncomp=1, dtype=ctypes.c_int)
        self.stride = ctypes.c_int(0)
        with open('./lib/NeighbourMatrixSource.cpp') as fh:
            src = fh.read()
        with open('./lib/NeighbourMatrixSource.h') as fh:
            hsrc = fh.read()

        self._lib = build.simple_lib_creator(hsrc, src, "OMP_N_MATRIX")

    def update(self, positions):

        self.timer_update.start()

        self._update(positions)

        self.version_id += 1

        self.timer_update.pause()
        opt.PROFILE[
            self.__class__.__name__+':update('+str(self.cell_width)+')'
        ] = (self.timer_update.time())

    def _update(self, positions):
        assert self.cell_list.cell_list is not None, "cell list is not initialised"
        assert self.cell_list.cell_list.dtype is ctypes.c_int, "bad datatype"
        assert positions.dtype is ctypes.c_double, "bad datatype"
        assert self._domain.cell_array.dtype is ctypes.c_int, "dtype"
        assert self.cell_list.cell_reverse_lookup.dtype is ctypes.c_int, "dtype"

        n = positions.npart_local
        if self.ncount.ncomp < n:
            self.ncount = host.Array(ncomp=n, dtype=ctypes.c_int)
        needed_stride = self.cell_list.max_cell_contents_count*27
        if self.stride < needed_stride:
            self.stride = needed_stride
        if self.matrix.ncomp < n*self.stride:
            self.matrix = host.Array(ncomp=n*self.stride, dtype=ctypes.c_int)

        ret = self._lib(
            ctypes.c_int(n),
            positions.ctypes_data,
            self.cell_list.cell_list.ctypes_data,
            self.cell_list.offset,
            self.cell_list.cell_reverse_lookup.ctypes_data,
            self._domain.cell_array.ctypes_data,
            self.matrix.ctypes_data,
            self.ncount.ctypes_data,
            ctypes.c_int(self.stride),
            ctypes.c_double(self.cell_width)
        )

        assert ret == 0, "lib failed, return code: " + str(ret)














