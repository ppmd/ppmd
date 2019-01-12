from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *
from itertools import product
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double

from ppmd.coulomb.sph_harm import LocalExpEval


def test_c_sph_harm_1():
    rng = np.random.RandomState(9476213)
    N = 20
    L = 20
    ncomp = (L**2)*2

    lee = LocalExpEval(L-1)
    
    for tx in range(20):
        point = [0, 0, 0]
        point[0] = rng.uniform(0, 10)
        point[1] = rng.uniform(0, 2*pi)
        point[2] = rng.uniform(0, pi)

        moments = rng.uniform(size=ncomp)
        rec = lee(moments, point)
        rep = lee.py_compute_phi_local(moments, point)
        rel = abs(rep)
        assert abs(rec - rep) / rel < 10.**-12
