from __future__ import print_function, division

import ppmd.lib
import ppmd.opt
import ppmd.runtime
from ppmd.pairloop.neighbourlist_14cell import _LIB_SOURCES

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes as ct
import math, os

# package level imports
from ppmd import host, runtime, opt, lib.build

_LIB_SOURCES = os.path.join(os.path.dirname(__file__), 'lib/')

###############################################################################
# NeighbourList definition 27 cell version
###############################################################################

class NeighbourListNonN3(object):
    def __init__(self, list=None):

        # timer inits
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


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_long)

        _initial_factor = math.ceil(27. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))

        self.max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_long)
        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)

        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1


        self._neighbour_lib = ppmd.lib.build.lib_from_source(
            _LIB_SOURCES+'NeighbourListNonN3',
            'NeighbourListNonN3',
            {
                'SUB_REAL': self._positions.ctype,
                'SUB_INT': 'int',
                'SUB_LONG': 'long'
            }
        )

        self.domain_id = self._domain.version_id

    def check_lib_rebuild(self):
        """return true if lib needs remaking"""

        return self.domain_id < self._domain.version_id

    def update_if_required(self):
        if self.version_id < self.cell_list.version_id:
            self.update()

    def update(self):
        assert self.max_len is not None and \
               self.list is not None and \
               self._neighbour_lib is not None, \
            "Neighbourlist setup not ran, or failed."

        self.timer_update.start()

        self._update()

        self.version_id = self.cell_list.version_id

        self.timer_update.pause()
        opt.PROFILE[
            self.__class__.__name__+':update'
        ] = (self.timer_update.time())


    def _update(self, attempt=1):
        dtype = self._positions.dtype
        assert ct.c_int == self._domain.cell_array.dtype
        assert ct.c_int == self.cell_list.cell_list.dtype
        assert ct.c_int == self.cell_list.cell_reverse_lookup.dtype
        assert dtype == self.cell_width_squared.dtype
        assert ct.c_long == self.neighbour_starting_points.dtype
        assert ct.c_int == self.list.dtype
        assert ct.c_long == self.max_len.dtype
        assert ct.c_int == self._return_code.dtype

        if self.neighbour_starting_points.ncomp < self._n() + 1:
            self.neighbour_starting_points.realloc(self._n() + 1)
        if runtime.VERBOSE > 3:
            print("rank:", self._domain.comm.Get_rank(), "rebuilding neighbour list")

        _n = self.cell_list.cell_list.end - self._domain.cell_count
        self._neighbour_lib.execute_no_time(
            ct.c_int(self._n()),
            ct.c_int(_n),
            self._positions.ctypes_data,
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
                print("rank:",\
                    self._domain.comm.Get_rank(),\
                    "neighbour list resizing",\
                    "old",\
                    self.max_len[0],\
                    "new",\
                    2 * self.max_len[0])

            self.max_len[0] *= 2
            self.list.realloc(self.max_len[0])

            assert attempt < 20, "Tried to create neighbour list too many times."

            self._update(attempt + 1)