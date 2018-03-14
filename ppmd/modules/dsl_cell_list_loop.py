"""
Create the C/C++ to loop over a cell list
"""

from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes
from cgen import *


class DSLCellListIter(object):
    def __init__(self, list_sym, offset_sym):
        self.list_sym = list_sym
        self.offset_sym = offset_sym
    
    def __call__(self, iter_sym, cell_sym, module):
        f0 = Initializer(
                Value('INT64', iter_sym),
                self.list_sym+'['+self.offset_sym+'+'+cell_sym+']'
            )
        f1 = While(
            iter_sym+'>-1',
            Block((
                module,
                Line(iter_sym + '=' + self.list_sym + '['+iter_sym+'];'),
            ))
        )

        return Module((f0,f1))
