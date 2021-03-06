import numpy as np

import pytest
from ppmd import *

from ppmd.coulomb import mm
from ppmd.coulomb import lm
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import EwaldOrthoganalHalf

import math

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double

import ppmd
from ppmd.lib import build

import time

from mpi4py import MPI

MPISIZE = MPI.COMM_WORLD.size
MPIRANK = MPI.COMM_WORLD.rank

from itertools import product


from ppmd.coulomb.fmm_pbc import LongRangeMTL
from ppmd.coulomb.direct import *

@pytest.mark.parametrize("MM_LM", (mm.PyMM, lm.PyLM))
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
@pytest.mark.parametrize("R", (3, 4))
def test_mm_lm_1(MM_LM, BC, R):
    

    N = 10000
    e = 10.
    L = 16


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)

    if BC == 'pbc':
        bias = np.sum(qi) / N
        qi -= bias
        assert abs(np.sum(qi)) < 10.**-12




    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi,
                A.Q: qi,
            })

    
    MM = MM_LM(A.P, A.Q, A.domain, BC, R, L)

    fmm_bc = {
        'free_space': True,
        '27': '27',
        'pbc': False
    }[BC]

    fmm = PyFMM(A.domain, r=R, l=L, free_space=fmm_bc)
    

    t0c = time.time()
    energy_to_test = MM(A.P, A.Q)
    t1c = time.time()
    

    t0f = time.time()
    # energy_fmm = fmm(A.P, A.Q)
    t1f = time.time()


    if BC == 'free_space':
        DFS = FreeSpaceDirect()
    elif BC == '27':
        DFS = NearestDirect(e)


    if MPISIZE > 1:
        correct = fmm(A.P, A.Q)
        t0 = 0.0
        t1 = 0.0
    else:
        if not BC == 'pbc':
            t0 = time.time()
            correct = DFS(N, A.P.view, A.Q.view)
            t1 = time.time()
        else:
            correct = fmm(A.P, A.Q)


    err = abs(energy_to_test - correct) / abs(correct)
    assert err < 10.**-6
    
    return
    if MPIRANK == 0:
        #print(err, err_fmm, energy_to_test, energy_fmm, correct)
        print("Direct", t1 - t0, MM_LM, t1c - t0c, "FMM", t1f - t0f)



@pytest.mark.parametrize("MM_LM", (mm.PyMM, lm.PyLM))
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
@pytest.mark.parametrize("R", (3, 4, 5))
def test_mm_lm_2(MM_LM, BC, R):

    if MPISIZE > 1 and R < 4:
        return
    elif MPISIZE > 8 and R < 5:
        return

    

    N = 100
    e = (50., 40., 30.)
    L = 16


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=e)
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)
    A.F = data.ParticleDat(ncomp=3)

    
    pi = np.zeros((N, 3), REAL)
    for dx in (0, 1, 2):
        pi[:, dx] = rng.uniform(low=-0.5*e[dx], high=0.5*e[dx], size=N)

    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)

    if BC == 'pbc':
        bias = np.sum(qi) / N
        qi -= bias
        assert abs(np.sum(qi)) < 10.**-12




    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi,
                A.Q: qi,
            })

    
    MM = MM_LM(A.P, A.Q, A.domain, BC, R, L)

    fmm_bc = {
        'free_space': True,
        '27': '27',
        'pbc': False
    }[BC]

    

    t0c = time.time()
    energy_to_test = MM(A.P, A.Q)
    t1c = time.time()
    




    if BC == 'free_space':
        DFS = FreeSpaceDirect()
    elif BC == '27':
        DFS = NearestDirect(e, tuples=MM.il[1])
    elif BC == 'pbc':
        DFS = PBCDirect(e, A.domain, L)
        EWALD = EwaldOrthoganalHalf(A.domain, 10.**-10, real_cutoff=e[0]/4, shared_memory='omp')


    correct = DFS(N, pi, qi)

    if BC == 'pbc':
        ec = EWALD(A.P, A.Q, A.F)
        err = abs(correct - ec)/ abs(ec)
        assert err < 10.**-5


    err = abs(energy_to_test - correct) / abs(correct)
    assert err < 10.**-6
    




