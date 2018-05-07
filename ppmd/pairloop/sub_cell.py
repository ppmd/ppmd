"""
Stencil tools for sub cell pair looping.
"""

from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import numpy as np
from itertools import product
from math import ceil
import ctypes

INT64 = ctypes.c_int64

class SubCell(object):
    """
    Constructs the interaction stencil for sub cell pairwise interactions
    where domains are decomposed into cells smaller than the interaction
    radius. Initalised with a 3-tuple holding the ratios between the smaller
    cell edge lengths and the interaction length.
    """
    def __init__(self, ratios=(0.5, 0.5, 0.5)):
       
        for cx in range(3):
            if ratios[cx] <= 0 or ratios[cx] > 1:
                raise RuntimeError('bad ratios passed')

        self.ratios = ratios
        ns = [int(ceil(1.0/ri)) for ri in ratios]
        
        quad = [(0,0,0)]
        # get the quadrants
        for tx in product(range(ns[0]), range(ns[1]), range(ns[2])):
            if  self._is_needed(tx):
                quad += self._mirror_quad(tx)
        
        # planes
        # xy
        for tx in product(range(ns[0]), range(ns[1]), [0]):
            if self._is_needed(tx):
                stx = (tx[0]+1, tx[1]+1)
                quad += [
                    ( stx[0],  stx[1], 0),
                    (-stx[0],  stx[1], 0),
                    (-stx[0], -stx[1], 0),
                    ( stx[0], -stx[1], 0)
                ]
        # yz
        for tx in product([0], range(ns[1]), range(ns[2])):
            if self._is_needed(tx):
                stx = (tx[1]+1, tx[2]+1)
                quad += [
                    (0,  stx[0],  stx[1]),
                    (0, -stx[0],  stx[1]),
                    (0, -stx[0], -stx[1]),
                    (0,  stx[0], -stx[1])
                ]
        # xz
        for tx in product(range(ns[0]), [0], range(ns[2])):
            if self._is_needed(tx):
                stx = (tx[0]+1, tx[2]+1)
                quad += [
                    ( stx[0], 0,  stx[1]),
                    (-stx[0], 0,  stx[1]),
                    (-stx[0], 0, -stx[1]),
                    ( stx[0], 0, -stx[1])
                ]

        # axis
        # x
        for tx in range(ns[0]):
            if self._is_needed((tx, 0, 0)):
                quad += [
                    ( tx+1, 0, 0),
                    (-tx-1, 0, 0)
                ]
        # y
        for tx in range(ns[1]):
            if self._is_needed((0, tx, 0)):
                quad += [
                    (0,  tx+1, 0),
                    (0, -tx-1, 0)
                ]
        # z
        for tx in range(ns[2]):
            if self._is_needed((0, 0, tx)):
                quad += [
                    (0, 0,  tx+1),
                    (0, 0, -tx-1)
                ]
        self.offsets = quad
        self.num_offsets = len(quad)
        self._offsets = np.zeros(self.num_offsets, dtype=INT64)

    def get_offsets(self, cell_array):
        """
        Given a cell array computes and returns the ctypes.c_int64 array
        of offsets for the passed cell array.
        """
        for oxi, ox in enumerate(self.offsets):
            self._offsets[oxi] = ox[0] + cell_array[0] * \
                (ox[1] + cell_array[1] * ox[2])
        return self._offsets

    def _is_needed(self, tx):
            return  (   (tx[0] * self.ratios[0])**2. + \
                        (tx[1] * self.ratios[1])**2. + \
                        (tx[2] * self.ratios[2])**2. 
                ) <= 1

    @staticmethod
    def _mirror_quad(tup):
        # shift the tuple into the quadrant
        st = [ti+1 for ti in tup]
        # mirror the offset into the other
        # 7 quadrants
        return [
            (   st[0],    st[1],    st[2]),
            (-1*st[0],    st[1],    st[2]),
            (-1*st[0], -1*st[1],    st[2]),
            (   st[0], -1*st[1],    st[2]),
            (   st[0],    st[1], -1*st[2]),
            (-1*st[0],    st[1], -1*st[2]),
            (-1*st[0], -1*st[1], -1*st[2]),
            (   st[0], -1*st[1], -1*st[2])
        ]



