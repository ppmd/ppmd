#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md
from ppmd.access import *

N = 1000
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
GlobalArray = md.data.GlobalArray
State = md.state.State


seed = 235236
rng = np.random.RandomState(seed=seed)


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

    A.u = GlobalArray(size=1)

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

    A.u = GlobalArray(size=1)

    return A

@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param

def test_host_looping_1(s_nd):
    """
    looping on non spatially decomposed state
    """

    print "THREADS", md.runtime.NUM_THREADS
    gid_sum = GlobalArray(size=1, dtype=ctypes.c_int, shared_memory='thread')

    pi = rng.uniform(-1*Eo2, Eo2, [N,3])
    vi = rng.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    s_nd.p[:] = pi
    s_nd.v[:] = vi
    s_nd.f[:] = fi
    s_nd.gid[:,0] = gidi

    kernel_code = '''
    V.i[0] = P.i[0];
    V.i[1] = P.i[1];
    V.i[2] = P.i[2];

    P.i[0] = (double) G.i[0];
    P.i[1] = (double) G.i[0];
    P.i[2] = (double) G.i[0];

    F.i[0] += 1.0;
    F.i[1] += 2.0;
    F.i[2] += 3.0;
    
    gid_sum[0] += G.i[0];
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': s_nd.p(md.access.RW),
                  'V': s_nd.v(md.access.W),
                  'F': s_nd.f(md.access.RW),
                  'G': s_nd.gid(md.access.R),
                  'gid_sum': gid_sum(INC_ZERO)}

    loop = md.loop.ParticleLoopOMP(kernel=kernel, dat_dict=kernel_map)
    loop.execute(n=N)

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

    assert gid_sum[0] == sum(range(N))*nproc, "critical GlobalArray Failure"



def test_host_looping_2(s_nd):
    """
    looping on non spatially decomposed state
    """

    print "THREADS", md.runtime.NUM_THREADS
    gid_sum = GlobalArray(size=1, dtype=ctypes.c_double, shared_memory='thread')

    pi = rng.uniform(-1*Eo2, Eo2, [N,3])

    vi = rng.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    s_nd.p[:] = pi
    s_nd.v[:] = vi
    s_nd.f[:] = fi
    s_nd.gid[:,0] = gidi

    kernel_code = '''

    double x[4] = {P.i[0], P.i[0], P.i[1], P.i[2]};
    for (int ix=0 ; ix<4 ; ix++)
    {
        gid_sum[0] += sin(x[ix]);
    }
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': s_nd.p(md.access.RW),
                  'V': s_nd.v(md.access.W),
                  'F': s_nd.f(md.access.RW),
                  'G': s_nd.gid(md.access.R),
                  'gid_sum': gid_sum(INC_ZERO)}

    loop = md.loop.ParticleLoopOMP(kernel=kernel, dat_dict=kernel_map)
    loop.execute(n=N)

    # relies on floating point equality

    assert abs(gid_sum[0] - np.sum(
        2.*np.sin(s_nd.p[0:N:,0]) + np.sin(s_nd.p[0:N:, 1]) + np.sin(s_nd.p[0:N:,2])
    )*nproc) < 10.**-10, "critical GlobalArray Failure"


def test_host_looping_3(s_nd):
    """
    looping on non spatially decomposed state
    """

    print "THREADS", md.runtime.NUM_THREADS
    gid_sum = GlobalArray(size=1, dtype=ctypes.c_double, shared_memory='thread')

    pi = rng.uniform(-1*Eo2, Eo2, [N,3])
    vi = rng.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    s_nd.p[:] = pi
    s_nd.v[:] = vi
    s_nd.f[:] = fi
    s_nd.gid[:,0] = gidi

    kernel_code = '''
    P.i[0] += 0.5*V.i[0];
    '''

    kernel = md.kernel.Kernel('test_host_looping_1',code=kernel_code)
    kernel_map = {'P': s_nd.p(md.access.RW),
                  'V': s_nd.v(md.access.W),
                  'F': s_nd.f(md.access.RW),
                  'G': s_nd.gid(md.access.R),
                  'gid_sum': gid_sum(INC_ZERO)}

    loop = md.loop.ParticleLoopOMP(kernel=kernel, dat_dict=kernel_map)
    loop.execute(n=N)


