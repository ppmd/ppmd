#!/usr/bin/python

import pytest
import ctypes
import numpy as np

from ppmd import *
from ppmd.coulomb.octal import *
from ppmd.coulomb.fmm import *

from ppmd.cuda import CUDA_IMPORT

if CUDA_IMPORT:
    from ppmd.cuda import *

cuda = pytest.mark.skipif("CUDA_IMPORT is False")

MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
MPISIZE = mpi.MPI.COMM_WORLD.Get_size()

@cuda
def test_cuda_fmm_1():
    R = 3

    crN = 10
    N = crN**3

    E = 3.*crN

    rc = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-6

    ASYNC = False
    free_space = True
    CUDA = True

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda=CUDA)


