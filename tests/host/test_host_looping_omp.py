#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md
from ppmd.access import *

Kernel = md.kernel.Kernel

N = 1000
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
GlobalArray = md.data.GlobalArray
ScalarArray = md.data.ScalarArray
State = md.state.State


seed = 235236
rng = np.random.RandomState(seed=seed)

N2 = 1000

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

@pytest.fixture(scope="module", params=(0, nproc-1))
def base_rank(request):
    return request.param

def test_host_looping_1(s_nd):
    """
    looping on non spatially decomposed state
    """

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

    assert abs(gid_sum[0] - np.sum(
        2.*np.sin(s_nd.p[0:N:,0]) + np.sin(s_nd.p[0:N:, 1]) + np.sin(s_nd.p[0:N:,2])
    )*nproc) < 10.**-10, "critical GlobalArray Failure"


def test_host_looping_3(s_nd):
    """
    looping on non spatially decomposed state
    """

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



def test_host_looping_4(state):


    pi = rng.uniform(-1*Eo2, Eo2, [N,3])

    data = rng.uniform(0, 1, [N, N2])
    state.data = ParticleDat(ncomp=N2)

    correct = np.zeros(N2)
    for nx in range(N2):
        correct[nx] = np.sum(data[:, nx])

    state.data[:, :] = data[:,:]
    state.p[:] = pi

    state.scatter_data_from(0)

    gdata = GlobalArray(size=N2, dtype=ctypes.c_double, shared_memory='thread')

    kernel_code = '''
    for(int ix=0 ; ix<%(N2)s ; ix++){
        gdata[ix] += data.i[ix];
    }
    ''' % {'N2':str(N2)}

    kernel = md.kernel.Kernel('test_looping_omp_4',code=kernel_code)
    kernel_map = {'data': state.data(md.access.R),
                  'gdata': gdata(INC_ZERO)}

    loop = md.loop.ParticleLoopOMP(kernel=kernel, dat_dict=kernel_map)
    loop.execute()

    for nx in range(N2):
        assert abs(correct[nx] - gdata[nx]) < 10.**-8, "INDEX: " + str(nx)



@pytest.fixture(scope="module", params=(ctypes.c_int, ctypes.c_double))
def DTYPE(request):
    return request.param


def test_host_looping_5(DTYPE):
    """
    Distributed looping without gather
    """
    # (READ, WRITE, INC_ZERO, INC, RW)

    PR = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    PW = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    PI0 = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    PI = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    PRW = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    SR = ScalarArray(ncomp=1, dtype=DTYPE)

    kernel_code = '''
    PW.i[0] = A1;
    PRW.i[0] = A1;
    PI0.i[0] += PR.i[0];
    PI.i[0] += SR[0];
    '''

    kernel = Kernel(
        'test_host_loop_dtype_access',
        kernel_code,
        static_args={
            'A1': DTYPE
        }
    )

    loop = md.loop.ParticleLoopOMP(
        kernel,
        dat_dict={
            'PR' :PR(READ),
            'PW' :PW(WRITE) ,
            'PI0':PI0(INC_ZERO),
            'PI' :PI(INC),
            'PRW':PRW(RW),
            'SR' :SR(READ)
        }
    )

    A1i = np.random.uniform(low=10, high=20, size=1)
    PRi = np.random.uniform(low=10, high=20, size=(N,1))
    PRWi = np.random.uniform(low=10, high=20, size=(N,1))
    SRi = np.random.uniform(low=10, high=20, size=1)
    PI0i = np.random.uniform(low=10, high=20, size=1)
    PIi = np.random.uniform(low=10, high=20, size=1)


    PR[:] = PRi[:]
    PRW[:] = PRWi[:]
    PW[:] = 0
    PI0[:] = PI0i[0]
    PI[:] = PIi[0]
    SR[:] = SRi[:]


    if DTYPE is ctypes.c_int:
        cast = int
    elif DTYPE is ctypes.c_double:
        cast = float
    else:
        raise RuntimeError

    a1 = cast(A1i[0])

    loop.execute(n=N, static_args={'A1': a1})

    for px in range(N):
        assert PR[px, 0] == cast(PRi[px, 0]), "read only data has changed"
        assert PW[px, 0] == a1, "bad write only particle dat"
        assert PRW[px, 0] == a1, "bad read/write particle dat"
        assert PI[px, 0] == SR[0] + cast(PIi[0]), "bad increment"
        assert PI0[px, 0] == PR[px, 0], "bad zero increment"

    assert SR[0] == cast(SRi[0]), "read only data has changed"

