from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np

# package level
from ppmd.lib import build


class ParticleDatModifier:

    def __init__(self, dat, is_positiondat):
        self.dat = dat
        self.is_positiondat = is_positiondat

    def __enter__(self):
        return self.dat.view

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dat.mark_halos_old()
        if self.is_positiondat:
            self.dat.group.invalidate_lists = True
            # need to trigger MPI rank consistency/ownership here
        








