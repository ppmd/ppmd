#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import math


import ppmd as md
import ppmd.cuda as mdc
from ppmd.access import *
Kernel = md.kernel.Kernel


cuda = pytest.mark.skipif("mdc.CUDA_IMPORT is False")

N = 1000
crN = 10 #cubert(N)
E = 8.

Eo2 = E/2.
tol = 0.1


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
    if mdc.CUDA_IMPORT_ERROR is not None:
        print(mdc.CUDA_IMPORT_ERROR)

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
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)
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
    A.nc = h_ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.u.halo_aware = True

    return A


@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param

@cuda
def test_cuda_pair_loop_1(state):
    """
    Set a cutoff slightly smaller than the smallest distance in the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()


    kernel_code = '''
    NC(0,0)+=1;
    '''

    kernel = md.kernel.Kernel('test_cuda_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNS(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 0


@cuda
def test_cuda_pair_loop_2(state):
    """
    Set a cutoff slightly larger than the smallest distance in the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()


    kernel_code = '''
    NC(0,0)+=1;
    '''

    kernel = md.kernel.Kernel('test_cuda_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNS(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6

@cuda
def test_cuda_pair_loop_3(state):
    """
    Set a cutoff slightly smaller than the next nearest neighbour distance in
    the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()


    kernel_code = '''
    NC(0,0)+=1;
    '''

    kernel = md.kernel.Kernel('test_cuda_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNS(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(2.)*cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6

@cuda
def test_cuda_pair_loop_4(state):
    """
    Set a cutoff slightly larger than the next nearest neighbour distance in
    the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()


    kernel_code = '''
    NC(0,0)+=1;
    '''

    kernel = md.kernel.Kernel('test_cuda_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNS(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(2.)*cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 18



@cuda
def test_cuda_pair_loop_5(state):
    """
    Set a cutoff slightly smaller than the 3rd nearest neighbour distance in
    the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()


    kernel_code = '''
    NC(0,0)+=1;
    '''

    kernel = md.kernel.Kernel('test_cuda_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNS(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(3.)*cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 18



@cuda
def test_cuda_pair_loop_ns_1(state):
    """
    Set a cutoff slightly larger than the 3rd nearest neighbour distance in
    the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()


    kernel_code = '''
    NC(0,0)+=1;
    '''

    kernel = md.kernel.Kernel('test_cuda_pair_loop_NS_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNS(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(3.)*cell_width+tol)

    state.nc.zero()

    loop.execute()
    nc = state.nc[:]
    for ix in range(state.npart_local):
        assert nc[ix] == 26


PairLoop = mdc.cuda_pairloop.PairLoopNeighbourListNS
@pytest.fixture(scope="module", params=list({ctypes.c_int, ctypes.c_double}))
def DTYPE(request):
    return request.param

def test_host_pair_loop_NS_dtypes_access(DTYPE):
    """
    Set a cutoff slightly larger than the 3rd nearest neighbour distance in
    the grid
    """

    A = State()

    crN2 = 10
    N = (crN2**3)*4
    A.npart = N

    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    cell_width = (0.5*float(E))/float(crN2)

    A.P = PositionDat(ncomp=3)
    A.P[:] = md.utility.lattice.fcc((crN2, crN2, crN2), (E, E, E))

    A.npart_local = (crN2**3)*4

    A.PR = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    A.PW = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    A.PI0 = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    A.PI = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    A.PRW = ParticleDat(npart=N, ncomp=1, dtype=DTYPE)
    A.SR = ScalarArray(ncomp=1, dtype=DTYPE)
    A.SI = ScalarArray(ncomp=1, dtype=DTYPE)
    A.SI0 = ScalarArray(ncomp=1, dtype=DTYPE)

    rng = np.random.RandomState(seed=2352413423)
    A1i = rng.uniform(low=10, high=20, size=1)
    PRi = rng.uniform(low=10, high=20, size=(N,1))
    PRWi = rng.uniform(low=10, high=20, size=(N,1))
    SRi = rng.uniform(low=10, high=20, size=1)
    PI0i = rng.uniform(low=10, high=20, size=1)
    PIi = rng.uniform(low=10, high=20, size=1)
    SIi = rng.uniform(low=10, high=20, size=1)
    SI0i = rng.uniform(low=10, high=20, size=1)

    A.PR[:] = PRi[0]
    A.PRW[:] = PRWi[:]
    A.PW[:] = 0
    A.PI0[:] = PI0i[0]
    A.PI[:] = PIi[0]
    A.SR[:] = SRi[0]
    A.SI[:] = SIi[0]
    A.SI0[:] = SI0i[0]

    A.filter_on_domain_boundary()

    kernel_code = '''
    PW.i[0] = A1;
    PRW.i[0] = A1;
    PI0.i[0] += PR.i[0];
    PI.i[0] += SR[0];
    SI[0] += A1;
    SI0[0] += A1;
    '''
    kernel = md.kernel.Kernel(
        'test_host_pair_loop_NS_dtype',
        code=kernel_code,
        static_args={
            'A1': DTYPE
        }
    )
    kernel_map = {
        'P': A.P(READ),
        'PR' :A.PR(READ),
        'PW' :A.PW(WRITE) ,
        'PI0':A.PI0(INC_ZERO),
        'PI' :A.PI(INC),
        'PRW':A.PRW(RW),
        'SR' :A.SR(READ),
        'SI' :A.SI(INC),
        'SI0':A.SI0(INC_ZERO)
    }

    loop = PairLoop(
        kernel=kernel,
        dat_dict=kernel_map,
        shell_cutoff=math.sqrt(2.)*cell_width+tol
    )

    if DTYPE is ctypes.c_int:
        cast = int
    elif DTYPE is ctypes.c_double:
        cast = float
    else:
        raise RuntimeError
    a1 = cast(A1i[0])

    loop.execute(static_args={'A1': a1})
    assert A.SR[0] == cast(SRi[0]), "read only data has changed"
    assert abs(A.SI[0] - (cast(SIi[0]) + A.npart_local*12*a1)) < 10.**-6, "bad scalar array INC"
    assert abs(A.SI0[0] - (12*A.npart_local*a1)) < 10.**-6, "bad scalar array INC_ZERO"

    for px in range(A.npart_local):
        assert A.PR[px, 0] == cast(PRi[0, 0]), "read only data has changed"
        assert A.PW[px, 0] == a1, "bad write only particle dat"
        assert A.PRW[px, 0] == a1, "bad read/write particle dat"
        assert abs(A.PI[px, 0] - 12*A.SR[0] - cast(PIi[0]))<10.**-14, "bad increment"
        assert abs(A.PI0[px, 0] - 12*A.PR[px, 0])<10.**-13, "bad zero increment"











