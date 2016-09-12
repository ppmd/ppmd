#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md

N = 1000
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
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

@pytest.fixture
def s_nd():
    """
    State with no domain, hence will not spatially decompose
    """

    A = State()
    A.npart = N
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

def test_host_looping_1(s_nd):
    """
    looping on non spatially decomposed state
    """

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    s_nd.p[:] = pi
    s_nd.v[:] = vi
    s_nd.f[:] = fi
    s_nd.gid[:,0] = gidi

    kernel_code = '''
    V(0) = P(0);
    V(1) = P(1);
    V(2) = P(2);

    P(0) = (double) G(0);
    P(1) = (double) G(0);
    P(2) = (double) G(0);

    F(0) += 1.0;
    F(1) += 2.0;
    F(2) += 3.0;
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': s_nd.p(md.access.RW),
                  'V': s_nd.v(md.access.W),
                  'F': s_nd.f(md.access.RW),
                  'G': s_nd.gid(md.access.R)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=N)

    assert np.sum(s_nd.v[:] == pi) == N*3

    # relies on floating point equality
    assert np.sum(s_nd.p[:,0] == np.array(gidi, dtype=ctypes.c_double)) == N
    assert np.sum(s_nd.p[:,1] == np.array(gidi, dtype=ctypes.c_double)) == N
    assert np.sum(s_nd.p[:,2] == np.array(gidi, dtype=ctypes.c_double)) == N

    # relies on floating point equality
    assert np.sum(s_nd.f[:,0] == np.ones([N]) ) == N
    assert np.sum(s_nd.f[:,1] == 2.0*np.ones([N]) ) == N
    assert np.sum(s_nd.f[:,2] == 3.0*np.ones([N]) ) == N

    # check gids are unaltered
    assert np.sum(s_nd.gid[:,0] == gidi) == N

def test_host_looping_2(state):
    """
    looping on spatially decomposed state
    """

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    state.p[:] = pi
    state.v[:] = vi
    state.f[:] = fi
    state.gid[:,0] = gidi

    kernel_code = '''
    V(0) = P(0);
    V(1) = P(1);
    V(2) = P(2);

    P(0) = (double) G(0);
    P(1) = (double) G(0);
    P(2) = (double) G(0);

    F(0) += 1.0;
    F(1) += 2.0;
    F(2) += 3.0;
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW),
                  'V': state.v(md.access.W),
                  'F': state.f(md.access.RW),
                  'G': state.gid(md.access.R)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=N)

    assert np.sum(state.v[:] == pi) == N*3

    # relies on floating point equality
    assert np.sum(state.p[:,0] == np.array(gidi, dtype=ctypes.c_double)) == N
    assert np.sum(state.p[:,1] == np.array(gidi, dtype=ctypes.c_double)) == N
    assert np.sum(state.p[:,2] == np.array(gidi, dtype=ctypes.c_double)) == N

    # relies on floating point equality
    assert np.sum(state.f[:,0] == np.ones([N]) ) == N
    assert np.sum(state.f[:,1] == 2.0*np.ones([N]) ) == N
    assert np.sum(state.f[:,2] == 3.0*np.ones([N]) ) == N

    # check gids are unaltered
    assert np.sum(state.gid[:,0] == gidi) == N



def test_host_looping_3(state, base_rank):
    """
    Distributed looping
    """

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    state.p[:] = pi
    state.v[:] = vi
    state.f[:] = fi
    state.gid[:,0] = gidi

    state.scatter_data_from(base_rank)


    kernel_code = '''
    V(0) = P(0);
    V(1) = P(1);
    V(2) = P(2);

    P(0) = (double) G(0);
    P(1) = (double) G(0);
    P(2) = (double) G(0);

    F(0) += 1.0;
    F(1) += 2.0;
    F(2) += 3.0;
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW),
                  'V': state.v(md.access.W),
                  'F': state.f(md.access.RW),
                  'G': state.gid(md.access.R)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)

    state.gather_data_on(base_rank)

    if rank == base_rank:

        # sort
        inds = state.gid[:,0].argsort()
        state.gid[:] = state.gid[inds]
        state.p[:] = state.p[inds]
        state.v[:] = state.v[inds]
        state.f[:] = state.f[inds]

        assert np.sum(state.v[:] == pi) == N*3

        # relies on floating point equality
        assert np.sum(state.p[:,0] == np.array(gidi, dtype=ctypes.c_double)) == N
        assert np.sum(state.p[:,1] == np.array(gidi, dtype=ctypes.c_double)) == N
        assert np.sum(state.p[:,2] == np.array(gidi, dtype=ctypes.c_double)) == N

        # relies on floating point equality
        assert np.sum(state.f[:,0] == np.ones([N]) ) == N
        assert np.sum(state.f[:,1] == 2.0*np.ones([N]) ) == N
        assert np.sum(state.f[:,2] == 3.0*np.ones([N]) ) == N

        # check gids are unaltered
        assert np.sum(state.gid[:,0] == gidi) == N



def test_host_looping_4(state, base_rank):
    """
    Distributed looping applied twice
    """

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    state.p[:] = pi
    state.v[:] = vi
    state.f[:] = fi
    state.gid[:,0] = gidi

    state.scatter_data_from(base_rank)


    kernel_code = '''
    V(0) = P(0);
    V(1) = P(1);
    V(2) = P(2);

    P(0) = (double) G(0);
    P(1) = (double) G(0);
    P(2) = (double) G(0);

    F(0) += 1.0;
    F(1) += 2.0;
    F(2) += 3.0;
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW),
                  'V': state.v(md.access.W),
                  'F': state.f(md.access.RW),
                  'G': state.gid(md.access.R)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)
    loop.execute(n=state.npart_local)

    state.gather_data_on(base_rank)

    if rank == base_rank:

        # sort
        inds = state.gid[:,0].argsort()
        state.gid[:] = state.gid[inds]
        state.p[:] = state.p[inds]
        state.v[:] = state.v[inds]
        state.f[:] = state.f[inds]

        # relies on floating point equality
        assert np.sum(state.v[:,0] == np.array(gidi, dtype=ctypes.c_double)) == N
        assert np.sum(state.v[:,1] == np.array(gidi, dtype=ctypes.c_double)) == N
        assert np.sum(state.v[:,2] == np.array(gidi, dtype=ctypes.c_double)) == N

        # relies on floating point equality
        assert np.sum(state.p[:,0] == np.array(gidi, dtype=ctypes.c_double)) == N
        assert np.sum(state.p[:,1] == np.array(gidi, dtype=ctypes.c_double)) == N
        assert np.sum(state.p[:,2] == np.array(gidi, dtype=ctypes.c_double)) == N

        # relies on floating point equality
        assert np.sum(state.f[:,0] == 2.0*np.ones([N]) ) == N
        assert np.sum(state.f[:,1] == 4.0*np.ones([N]) ) == N
        assert np.sum(state.f[:,2] == 6.0*np.ones([N]) ) == N

        # check gids are unaltered
        assert np.sum(state.gid[:,0] == gidi) == N


def test_host_looping_5(state, base_rank):
    """
    Distributed looping without gather
    """

    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    state.p[:] = pi
    state.v[:] = vi
    state.f[:] = fi
    state.gid[:,0] = gidi

    state.scatter_data_from(base_rank)


    kernel_code = '''
    V(0) = P(0);
    V(1) = P(1);
    V(2) = P(2);

    P(0) = (double) G(0);
    P(1) = (double) G(0);
    P(2) = (double) G(0);

    F(0) += 1.0;
    F(1) += 2.0;
    F(2) += 3.0;
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW),
                  'V': state.v(md.access.W),
                  'F': state.f(md.access.RW),
                  'G': state.gid(md.access.R)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)

    M = state.npart_local

    # relies on floating point equality
    assert np.sum(state.p[:M:,0] == np.array(state.gid[:M:], dtype=ctypes.c_double)) == M
    assert np.sum(state.p[:M:,1] == np.array(state.gid[:M:], dtype=ctypes.c_double)) == M
    assert np.sum(state.p[:M:,2] == np.array(state.gid[:M:], dtype=ctypes.c_double)) == M

    # relies on floating point equality
    assert np.sum(state.f[:M:,0] == np.ones([M]) ) == M
    assert np.sum(state.f[:M:,1] == 2.0*np.ones([M]) ) == M
    assert np.sum(state.f[:M:,2] == 3.0*np.ones([M]) ) == M









