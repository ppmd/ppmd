from __future__ import print_function, division

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes as ct
import math, os

# package level imports
from ppmd import host, runtime
import ppmd.lib.build

_LIB_SOURCES = os.path.join(os.path.dirname(__file__), 'lib/')

###############################################################################
# NeighbourListv2 definition 14 cell version
###############################################################################

class NeighbourListv2(object):
    def __init__(self, list=None):
        self.timer_update = ppmd.opt.Timer()

        self._cell_list_func = list
        self.cell_list = list
        self.max_len = None
        self.list = None
        self.lib = None

        self.domain_id = 0
        self.version_id = 0
        """Update tracking of neighbour list. """

        self.cell_width = None
        self.time = 0
        self._time_func = None


        self._positions = None
        self._domain = None
        self.neighbour_starting_points = None
        self.cell_width_squared = None
        self._neighbour_lib = None
        self._n = None

        self.n_local = None
        self.n_total = None

        self._last_n = -1
        """Return the number of particle that have neighbours listed"""
        self._return_code = None

    def update(self, _attempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        self.timer_update.start()

        if self.neighbour_starting_points.ncomp < self._n() + 1:
            # print "resizing"
            self.neighbour_starting_points.realloc(self._n() + 1)
        if runtime.VERBOSE > 3:
            print("rank:", self._domain.comm.Get_rank(),
                  "rebuilding neighbour list")

        assert ct.c_double == self._domain.boundary.dtype
        assert ct.c_double == self._positions.dtype
        assert ct.c_double == self._domain.cell_edge_lengths.dtype
        assert ct.c_int == self._domain.cell_array.dtype
        assert ct.c_int == self.cell_list.cell_list.dtype
        assert ct.c_int == self.cell_list.cell_reverse_lookup.dtype
        assert ct.c_double == self.cell_width_squared.dtype
        assert ct.c_long == self.neighbour_starting_points.dtype
        assert ct.c_int == self.list.dtype
        assert ct.c_long == self.max_len.dtype
        assert ct.c_int == self._return_code.dtype

        _n = self.cell_list.cell_list.end - self._domain.cell_count
        self._neighbour_lib(
            ct.c_int(self._n()),
            ct.c_int(_n),
            self._domain.boundary.ctypes_data,
            self._positions.ctypes_data,
            self._domain.cell_edge_lengths.ctypes_data,
            self._domain.cell_array.ctypes_data,
            self.cell_list.cell_list.ctypes_data,
            self.cell_list.cell_reverse_lookup.ctypes_data,
            self.cell_width_squared.ctypes_data,
            self.neighbour_starting_points.ctypes_data,
            self.list.ctypes_data,
            self.max_len.ctypes_data,
            self._return_code.ctypes_data
        )

        self.n_total = self._positions.npart_total
        self.n_local = self._n()
        self._last_n = self._n()


        if self._return_code[0] < 0:
            if runtime.VERBOSE > 2:
                print("rank:", self._domain.comm.Get_rank(),
                      "neighbour list resizing", "old", self.max_len[0], "new",
                      2*self.max_len[0])

            self.max_len[0] *= 2
            self.list.realloc(self.max_len[0])

            assert _attempt < 20, "Tried creating list too many times."

            self.update(_attempt + 1)
            return

        self.version_id = self.cell_list.version_id

        self.timer_update.pause()

    def setup(self, n, positions, domain, cell_width):

        assert self.cell_list.cell_list is not None, "No cell to particle " \
                                                     "map setup"

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2,
                                             dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        self.neighbour_starting_points = host.Array(ncomp=n() + 1,
                                                    dtype=ct.c_long)

        _n = n()
        if _n < 10:
            _n = 10

        _initial_factor = math.ceil(
            15. * (_n ** 2) / (domain.cell_array[0] * domain.cell_array[1] *
                               domain.cell_array[2])
        )

        if _initial_factor < 10:
            _initial_factor = 10

        self.max_len = host.Array(initial_value=_initial_factor,
                                  dtype=ct.c_long)

        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)

        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1

        self._neighbour_lib = ppmd.lib.build.lib_from_file_source(
            _LIB_SOURCES+'NeighbourListv2', 'NeighbourListv2',
            {
                'SUB_REAL': 'double',
                'SUB_INT': 'int',
                'SUB_LONG': 'long'
            }
        )['NeighbourListv2']


