from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier()
DEBUG = True


def test_fmm_init_1():


    E = 10.
    N = 1000

    A = state.State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)


    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))
    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias
    A.scatter_data_from(0)

    fmm = PyFMM(domain=A.domain, N=1000)
    fmm._compute_cube_contrib(A.P, A.Q)

    print(np.sum(fmm.entry_data[:]))

















