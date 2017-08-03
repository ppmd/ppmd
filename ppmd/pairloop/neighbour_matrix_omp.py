# system level
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes, os

from ppmd import host, runtime, opt
from ppmd.lib import build


################################################################################################################
# NeighbourList definition 27 cell version
################################################################################################################

class NeighbourListOMP(object):
    def __init__(self, n, positions, domain, cell_width, cell_list):
        self._n_func = n
        self._positions = positions
        self._domain = domain
        self.cell_width = cell_width
        self.cell_list = cell_list
        self.version_id = 0
        self.domain_id = 0
        self.n_local = None
        self.timer_update = opt.Timer(runtime.TIMER)
        self.matrix = host.Array(ncomp=1, dtype=ctypes.c_int)
        self.ncount = host.Array(ncomp=1, dtype=ctypes.c_int)
        self.stride = ctypes.c_int(0)
        self.total_num_neighbours = 0
        self.max_size = 0

        bn = os.path.join(os.path.dirname(__file__), 'lib')
        bn += '/NeighbourMatrixSource'
        self._lib = build.lib_from_source(bn, 'OMPNeighbourMatrix')

        self._lib.restype = ctypes.c_longlong

    def update_if_required(self):
        if self.version_id < self.cell_list.version_id or \
            self.domain_id < self._domain.version_id:
            self.update()


    def update(self):

        self.timer_update.start()

        self._update()

        self.version_id = self.cell_list.version_id
        self.domain_id = self._domain.version_id

        self.timer_update.pause()
        opt.PROFILE[
            self.__class__.__name__+':update('+str(self.cell_width)+')'
        ] = (self.timer_update.time())

        self.max_size = max(self.max_size, self.matrix.size)

        opt.PROFILE[
            self.__class__.__name__+':nbytes('+str(self.cell_width)+')'
        ] = (self.max_size)

    def _update(self):
        positions = self._positions
        assert self.cell_list.cell_list is not None, "cell list is not initialised"
        assert self.cell_list.cell_list.dtype is ctypes.c_int, "bad datatype"
        assert positions.dtype is ctypes.c_double, "bad datatype"
        assert self._domain.cell_array.dtype is ctypes.c_int, "dtype"
        assert self.cell_list.cell_reverse_lookup.dtype is ctypes.c_int, "dtype"
        n = self._n_func()
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
            ctypes.c_double(self.cell_width**2.0)
        )

        assert ret >= 0, "lib failed, return code: " + str(ret)
        self.n_local = n
        self.total_num_neighbours = ret













