from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import numpy as np
import ctypes
import os

from ppmd import runtime
from ppmd.lib.build import lib_from_file_source

REAL = ctypes.c_double
INT64 = ctypes.c_int64


_LIB_SOURCES = os.path.join(os.path.dirname(__file__), 'lib/plain_cell_list/')

class PlainCellList(object):
    """
    Provides basic cell list capabilities with padded regions
    for halo particles.
    """
    def __init__(self, cell_width, local_boundary ):
        assert cell_width > 0

        
        # local_boundary has layout xl, xh, yl, yh, zl, zh
        self._local_boundary = local_boundary
        self.boundary = np.zeros(6, dtype=REAL)
        
        lb = self._local_boundary
        extentx = lb[1] - lb[0]
        extenty = lb[3] - lb[2]
        extentz = lb[5] - lb[4]

        assert extentx > 0 and extenty > 0 and extentz > 0
        
        ncellx = int(extentx/cell_width)
        ncelly = int(extenty/cell_width)
        ncellz = int(extentz/cell_width)
        
        widthx = extentx/ncellx
        widthy = extenty/ncelly
        widthz = extentz/ncellz
        
        # lower boundary, needed for cell binning
        self._lboundary = np.zeros(3, dtype=REAL)
        self._lboundary[0] = lb[0] - 2*widthx
        self._lboundary[1] = lb[2] - 2*widthy
        self._lboundary[2] = lb[4] - 2*widthz
        # cell width, needed for cell binning
        self._inv_cell_widths = np.zeros(3, dtype=REAL)
        self._inv_cell_widths[0] = 1.0/widthx
        self._inv_cell_widths[1] = 1.0/widthy
        self._inv_cell_widths[2] = 1.0/widthz
        
        #cell array size
        self.cell_array = np.zeros(3, dtype=INT64)
        self.cell_array[0] = ncellx + 4
        self.cell_array[1] = ncelly + 4
        self.cell_array[2] = ncellz + 4

        self.cell_count = np.product(self.cell_array)
        
        # cell list
        self.list = np.zeros(100, dtype=INT64)
        # cell reverse lookup
        self.cell_reverse_lookup = np.zeros(100, dtype=INT64)
        # cell contents count
        self.cell_contents_count = np.zeros(self.cell_count, dtype=INT64)
        
        # offset to cell part of cell list
        self.cell_offset = INT64(-1)

        # max cell contents count
        self._max_count = 0


        with open(_LIB_SOURCES + 'PlainCellList.h') as fh:
            hpp = fh.read()
        with open(_LIB_SOURCES + 'PlainCellList.cpp') as fh:
            cpp = fh.read()

        self._lib = lib_from_file_source(_LIB_SOURCES + 'PlainCellList', 
                'PlainCellList',
                {})['PlainCellList']

    def _check_len(self, npart):
        n = npart + self.cell_count
        if self.list.shape[0] < n:
            self.list = np.zeros(n, dtype=INT64)
        self.cell_offset = self.list.shape[0] - self.cell_count
        if self.cell_reverse_lookup.shape[0] < n:
            self.cell_reverse_lookup = np.zeros(n, dtype=INT64)

    def sort(self, positions, npart):
        """
        Sort particles into cells
        :param positions: ParticleDat to use for positions.
        :param npart: Number of particles to sort.
        """
        self._check_len(npart)
        err = self._lib(
            positions.ctypes_data,
            INT64(npart),
            INT64(self.cell_offset),
            self.cell_array.ctypes.get_as_parameter(),
            self._inv_cell_widths.ctypes.get_as_parameter(),
            self._lboundary.ctypes.get_as_parameter(),
            self.list.ctypes.get_as_parameter(),
            self.cell_contents_count.ctypes.get_as_parameter(),
            self.cell_reverse_lookup.ctypes.get_as_parameter()
        )
        if err < 0:
            raise RuntimeError('PlainCellList returned negative error code: '\
                    + str(err))
        self._max_count = np.max(self.cell_contents_count)

    @property
    def max_cell_contents_count(self):
        return self._max_count




