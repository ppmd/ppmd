from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

import cProfile
import profile
import pstats
import tempfile

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'), filename)


from ppmd import *
from ppmd.coulomb import wigner

from ppmd.coulomb.fmm_pbc import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

def red(*input):
    try:
        from termcolor import colored
        return colored(*input, color='red')
    except Exception as e: return input
def green(*input):
    try:
        from termcolor import colored
        return colored(*input, color='green')
    except Exception as e: return input
def yellow(*input):
    try:
        from termcolor import colored
        return colored(*input, color='yellow')
    except Exception as e: return input

def red_tol(val, tol):
    if abs(val) > tol:
        return red(str(val))
    else:
        return green(str(val))



def test_fmm_shell_iterator_1():

    ii = shell_iterator(0)
    assert len(ii) == 1
    assert ii[0][0] == 0
    assert ii[0][1] == 0
    assert ii[0][2] == 0

    rx = 1
    nx = 2*rx+1
    ii = shell_iterator(rx)
    assert len(ii) == 26
    test_mat = np.zeros((nx,nx,nx))
    true_mat = np.ones((nx,nx,nx))
    test_mat[rx,rx,rx] = 1

    for ix in ii:
        test_mat[rx+ix[2], rx+ix[1], rx+ix[0]] += 1

    assert np.sum(np.abs(true_mat-test_mat)) == 0

    for rx in range(2,10):
        nx = 2*rx+1
        ii = shell_iterator(rx)
        test_mat = np.zeros((nx,nx,nx))
        true_mat = np.ones((nx,nx,nx))

        test_mat[1:-1:,1:-1,1:-1] = 1

        for ix in ii:
            test_mat[rx+ix[2], rx+ix[1], rx+ix[0]] += 1

        assert np.sum(np.abs(true_mat-test_mat)) == 0


    e=10.
    eps=10.0**-8
    l=30
    d = domain.BaseDomainHalo(extent=(e,e,e))
    G = FMMPbc(l, eps, d, ctypes.c_double)
    G.compute_f()




@pytest.mark.skipif("True")
def test_fmm_setup_time_1():

    R = 3
    eps = 10.**-8

    N = 32
    E = 4.

    #N = 10000
    #E = 100.

    rc = E/8


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    prof = cProfile.Profile()
    free_space = False
    prof.enable()
    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space, l=40)
    prof.disable()

    prof.create_stats()

    p = pstats.Stats(prof)

    p.sort_stats('cumulative').print_stats(10)


@pytest.mark.skipif("True")
def test_fmm_setup_time_2():

    R = 3
    eps = 10.**-8

    N = 32
    E = 4.

    #N = 10000
    #E = 100.

    rc = E/8


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    free_space = False
    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space, l=10)

















