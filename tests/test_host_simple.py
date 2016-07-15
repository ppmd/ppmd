#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md
from mpi4py import MPI

N = 16
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI_HANDLE.rank
nproc = md.mpi.MPI_HANDLE.nproc

PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
State = md.state.State

@pytest.fixture
def state():
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param


def test_host_npart(state):
    assert state.npart == N
    assert state.npart_local == 0


def test_host_broadcast_data_from(state, base_rank):


    state.p[:] = (rank+1)*np.ones([N,3])
    state.v[:] = (rank+1)*np.ones([N,3])
    state.f[:] = (rank+1)*np.ones([N,3])

    state.broadcast_data_from(base_rank)

    assert np.sum((base_rank+1)*np.ones([N,3]) == state.p[:]) == N*3
    assert np.sum((base_rank+1)*np.ones([N,3]) == state.v[:]) == N*3
    assert np.sum((base_rank+1)*np.ones([N,3]) == state.f[:]) == N*3
    assert state.npart_local == N



def test_host_particle_dat_copy(state):

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])

    #TODO automatically set local number of particles?
    state.npart_local = N

    state.p[:] = pi


    pp = state.p.copy()

    assert np.sum(pi[:] == state.p[:]) == N*3
    assert np.sum(pp[:] == state.p[:]) == N*3
    assert np.sum(pp[:] == pi[:]) == N*3


def test_host_particle_dat_copy_distributed(state, base_rank):
    """
    As above but distributed with a gather
    """

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    state.p[:] = pi
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(base_rank)


    pp = state.p.copy()
    gidp = state.gid.copy()

    pp.gather_data_on(base_rank)
    gidp.gather_data_on(base_rank)



    if rank == base_rank:
        inds = gidp[:,0].argsort()

        #print pp[:]
        #print 60*"-"

        pp[:] = pp[inds]

        assert np.sum(pp[:] == pi[:]) == N*3





def test_host_scatter_gather(state, base_rank):

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

    s = np.array([state.npart_local])
    r = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(s[0], r, MPI.SUM)

    # globally we should still have N particles
    assert r[0] == state.npart
    assert r[0] == N


    state.gather_data_on(base_rank)

    if rank == base_rank:
        inds = state.gid[:,0].argsort()

        sgid = state.gid[inds]
        sp = state.p[inds]
        sv = state.v[inds]

        # These equalities are floating point. No arithmetic should be done
        # but it is conceivable that it may happen.

        '''
        print "\n", state.gid[:,0]

        for ix in range(N):

            assert sgid[ix,:] == ix
            print "\n", sp[ix,:], pi[ix,:], np.abs( sp[ix,:] - pi[ix,:] )
            assert np.sum(np.abs( sp[ix,:] - pi[ix,:] )) < 10**(-14)
        '''

        #assert np.sum(sp == pi) == N*3
        #assert np.sum(sv == vi) == N*3
        assert np.sum(sgid == np.arange(N)) == N
        assert np.sum(sp == pi) == N*3
        assert np.sum(sv == vi) == N*3


