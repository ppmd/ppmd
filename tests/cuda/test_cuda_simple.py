#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import sys



import ppmd as md
import ppmd.cuda as mdc



cuda = pytest.mark.skipif("mdc.CUDA_IMPORT is False")


N = 1000
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI_HANDLE.rank
nproc = md.mpi.MPI_HANDLE.nproc

if mdc.CUDA_IMPORT:
    PositionDat = mdc.cuda_data.PositionDat
    ParticleDat = mdc.cuda_data.ParticleDat
    ScalarArray = mdc.cuda_data.ScalarArray
    State = mdc.cuda_state.State

@cuda
@pytest.fixture
def state(request):
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.p = PositionDat(ncomp=3)
    A.v = PositionDat(ncomp=3)
    A.f = PositionDat(ncomp=3)
    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True


    #request.addfinalizer(A.free_all)
    return A

@cuda
def test_npart(state):
    assert state.npart == N
    assert state.npart_local == 0

@cuda
def test_broadcast_data_from(state):

    state.p[:] = (rank+1)*np.ones([N,3])
    state.v[:] = (rank+1)*np.ones([N,3])
    state.f[:] = (rank+1)*np.ones([N,3])

    state.broadcast_data_from(0)

    assert np.sum(np.ones([N,3]) == state.p[:]) == N*3
    assert np.sum(np.ones([N,3]) == state.v[:]) == N*3
    assert np.sum(np.ones([N,3]) == state.f[:]) == N*3
    assert state.npart_local == N
    sys.stdout.flush()
    sys.stderr.flush()


@cuda
def test_scatter_gather(state):
    state.p[:] = np.random.uniform(-1*Eo2, Eo2, [N,3])
    state.v[:] = np.random.normal(0, 2, [N,3])
    state.f[:] = np.zeros([N,3])

    state.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    state.gid[:,0] = np.arange(N)

    state.scatter_data_from(0)
    state.gather_data_on(0)

    if rank == 0:
        ret = state.gid[:,0]
        ret.sort()
        ret_true = np.arange(state.npart)
        assert state.npart == np.sum(ret==ret_true)





