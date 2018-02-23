#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import math


from ppmd import *
from ppmd.access import *


N = 1000
crN = 10 #cubert(N)
E = 8.

Eo2 = E/2.

tol = 10.**(-12)

rank = mpi.MPI.COMM_WORLD.Get_rank()
nproc = mpi.MPI.COMM_WORLD.Get_size()


PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
GlobalArray = data.GlobalArray
State = state.State
PairLoop = pairloop.CellByCellOMP
Kernel = kernel.Kernel


@pytest.fixture
def state():
    A = State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = GlobalArray(ncomp=1)
    A.u.halo_aware = True

    return A


@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param


def test_init_1(state):
    """
    Set a cutoff slightly smaller than the smallest distance in the grid
    """

    cell_width = float(E)/float(crN)
    
    
    assert state.npart == N

    rng = np.random.RandomState(seed=1234)
    state.p[:] = rng.uniform(low=-Eo2, high=Eo2, size=(N,3))
    state.scatter_data_from(0)
    
    # check no particles were gained or lost
    npl = np.array((state.npart_local))
    m = mpi.all_reduce(npl)
    assert m == N


    sh = pairloop.state_handler.StateHandler(state, cell_width)

    # check no particles were gained or lost
    npl = np.array((state.npart_local))
    m = mpi.all_reduce(npl)
    assert m == N
    
    # no halo exchanges should have yet occured
    assert state.npart_halo == 0
    
    sh.pre_execute({
        'p': state.p(access.READ)
    })
    

    ctotals = np.sum(state.get_cell_to_particle_map().cell_contents_count[:])

    assert state.npart_halo + state.npart_local == ctotals














