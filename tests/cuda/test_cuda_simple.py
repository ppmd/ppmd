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

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()

if mdc.CUDA_IMPORT:
    PositionDat = mdc.cuda_data.PositionDat
    ParticleDat = mdc.cuda_data.ParticleDat
    ScalarArray = mdc.cuda_data.ScalarArray
    State = mdc.cuda_state.State


h_PositionDat = md.data.PositionDat
h_ParticleDat = md.data.ParticleDat
h_ScalarArray = md.data.ScalarArray
h_State = md.state.State




@cuda
@pytest.fixture
def state(request):
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = mdc.cuda_domain.BoundaryTypePeriodic()
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    return A



@cuda
@pytest.fixture
def h_state(request):
    A = h_State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    A.p = h_PositionDat(ncomp=3)
    A.v = h_ParticleDat(ncomp=3)
    A.f = h_ParticleDat(ncomp=3)
    A.gid = h_ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.u = h_ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A


@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param


@cuda
def test_cuda_npart(state):
    assert state.npart == N
    assert state.npart_local == 0

@cuda
def test_cuda_broadcast_data_from(state, base_rank):

    state.p[:] = (rank+1)*np.ones([N,3])
    state.v[:] = (rank+1)*np.ones([N,3])
    state.f[:] = (rank+1)*np.ones([N,3])

    state.broadcast_data_from(base_rank)

    # check device broadcast
    assert np.sum((base_rank+1)*np.ones([N,3]) == state.p[:]) == N*3
    assert np.sum((base_rank+1)*np.ones([N,3]) == state.v[:]) == N*3
    assert np.sum((base_rank+1)*np.ones([N,3]) == state.f[:]) == N*3
    assert state.npart_local == N



@cuda
def test_cuda_check_scatter(state, h_state, base_rank):

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])

    state.p[:] = pi
    state.v[:] = vi
    state.f[:] = fi
    state.gid[:,0] = np.arange(N)

    h_state.p[:] = pi
    h_state.v[:] = vi
    h_state.f[:] = fi
    h_state.gid[:,0] = np.arange(N)

    # check initialisation is the same
    assert np.sum(state.p[:] == h_state.p[:]) == N*3
    assert np.sum(state.v[:] == h_state.v[:]) == N*3
    assert np.sum(state.f[:] == h_state.f[:]) == N*3
    assert np.sum(state.gid[:] == h_state.gid[:]) == N

    # broadcast host state and cuda state
    state.broadcast_data_from(base_rank)
    h_state.broadcast_data_from(base_rank)

    # check scatter was identical
    assert np.sum(state.p[:] == h_state.p[:]) == h_state.npart_local*3
    assert np.sum(state.v[:] == h_state.v[:]) == h_state.npart_local*3
    assert np.sum(state.f[:] == h_state.f[:]) == h_state.npart_local*3
    assert np.sum(state.gid[:] == h_state.gid[:]) == h_state.npart_local


@cuda
def test_cuda_scatter_gather(state, base_rank):

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])

    state.p[:] = pi
    state.v[:] = vi
    state.f[:] = fi

    state.gid[:,0] = np.arange(N)

    #state.broadcast_data_from(0)
    #state.filter_on_domain_boundary()
    state.scatter_data_from(base_rank)

    state.gather_data_on(base_rank)

    if rank == base_rank:
        inds = state.gid[:,0].argsort()

        sgid = state.gid[inds]
        sp = state.p[inds]
        sv = state.v[inds]

        # These equalities are floating point. No arithmetic should be done
        # but it is conceivable that it may happen.
        assert np.sum(sgid == np.arange(N)) == N
        assert np.sum(sp == pi) == N*3
        assert np.sum(sv == vi) == N*3

@cuda
def test_cuda_1_norm():
    a = np.random.uniform(size=[N,3])

    ap = ParticleDat(initial_value=a, dtype=ctypes.c_double)
    ap.npart_local = N


    for ix in range(N):
        assert a[ix, 0] == ap[ix, 0]
        assert a[ix, 1] == ap[ix, 1]
        assert a[ix, 2] == ap[ix, 2]


    an = np.linalg.norm(a.reshape([N*3,1]), np.inf)
    apn = ap.norm_linf()


    assert abs(an - apn) < 10.** (-15)


@cuda
def test_cuda_plus_equal_1():
    A = ParticleDat(npart=1, ncomp=1, dtype=ctypes.c_int)

    A[0, 0] = 0
    rx = 0

    for ix in range(10):
        rx += ix
        A[0, 0] += ix
        assert rx == A[0, 0]

@cuda
def test_cuda_plus_equal_2():

    A = ParticleDat(npart=1, ncomp=3, dtype=ctypes.c_int)

    A[0, :] = 1
    rx = 1

    for ix in range(4, 10):
        rx += ix
        A[0, :] += ix

        assert rx == A[0, 0]
        assert rx == A[0, 1]
        assert rx == A[0, 2]












