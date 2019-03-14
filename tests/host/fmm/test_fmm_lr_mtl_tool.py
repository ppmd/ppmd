__author__ = "W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

#from ppmd_vis import plot_spheres


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.fmm_pbc import *
import time

from math import *

REAL = ctypes.c_double



def test_fmm_lr_mtl_tool_1():
    R = 3
    L = 4
    eps = 10.**-8
    free_space = True

    E = 4.
    
    rng = np.random.RandomState(seed=23058)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    fmm = PyFMM(domain=A.domain, r=R, l=L)
    lr_mtl = LongRangeMTL(L, A.domain)

    ncomp = 2*(L**2)

    for nx in range(20):
        M = np.zeros(ncomp, REAL)
        L1 = np.zeros(ncomp, REAL)
        L2 = np.zeros(ncomp, REAL)

        M[:] = rng.uniform(size=ncomp)

        fmm._lr_mtl_func(M, L1)
        fmm.dipole_corrector(M, L1)

        lr_mtl(M, L2)

        err = np.linalg.norm(L1 - L2, np.inf)
        assert err < 10.**-15
        
    fmm.free()

