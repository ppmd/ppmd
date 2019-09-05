from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np



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

from ppmd.coulomb.sph_harm import LocalExpEval, MultipoleExpCreator, MultipoleDotVecCreator


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

    for tx in range(20):
        point = [0.5,         1.57079633, -1.57079633]
        point[0] = rng.uniform(0, 10)
        point[1] = rng.uniform(0, 2*pi)
        point[2] = rng.uniform(0, pi)

        moments = rng.uniform(size=ncomp)
        rec = lee(moments, point)
        rep = lee.py_compute_phi_local(moments, point)
        rel = abs(rep)
        assert abs(rec - rep) / rel < 10.**-12


    for tx in range(20):
        point = [0.5,         1.57079633, 1.57079633]
        point[0] = rng.uniform(0, 10)
        point[1] = rng.uniform(0, 2*pi)
        point[2] = rng.uniform(0, pi)

        moments = rng.uniform(size=ncomp)
        rec = lee(moments, point)
        rep = lee.py_compute_phi_local(moments, point)
        rel = abs(rep)
        assert abs(rec - rep) / rel < 10.**-12





def test_c_sph_harm_2():
    rng = np.random.RandomState(9473)
    N = 20
    L = 20
    ncomp = (L**2)*2

    lee = MultipoleExpCreator(L-1)
    lee2 = MultipoleDotVecCreator(L-1)
    
    for tx in range(20):

        radius = rng.uniform()
        phi = rng.uniform(0, math.pi * 2)
        theta = rng.uniform(0, math.pi)

        sph = (radius, theta, phi)

        correct = np.zeros(ncomp, REAL)
        correctd = np.zeros(ncomp, REAL)
        to_test = np.zeros(ncomp, REAL)
        to_testm = np.zeros(ncomp, REAL)
        to_testd = np.zeros(ncomp, REAL)


        lee.multipole_exp(sph, 1.0, to_test)
        lee.py_multipole_exp(sph, 1.0, correct)
        lee2.dot_vec_multipole(sph, 1.0, to_testd, to_testm)

        err = np.linalg.norm(to_test - correct, np.inf)
        assert err < 10.**-14
        err = np.linalg.norm(to_testm - correct, np.inf)
        assert err < 10.**-14

        lee2.py_dot_vec(sph, 1.0, correctd)
        
        err = np.linalg.norm(to_testd - correctd, np.inf)
        assert err < 10.**-14








