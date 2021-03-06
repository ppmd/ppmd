#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import math


import ppmd as md
import ppmd.cuda as mdc


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



@pytest.mark.skip(reason="implementation broken")
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


@pytest.mark.skip(reason="implementation broken")
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

@pytest.mark.skip(reason="implementation broken")
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

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNSSplit(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 0

@pytest.mark.skip(reason="implementation broken")
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

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNSSplit(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6

@pytest.mark.skip(reason="implementation broken")
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

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNSSplit(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(2.)*cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6

@pytest.mark.skip(reason="implementation broken")
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

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNSSplit(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(2.)*cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 18


@pytest.mark.skip(reason="implementation broken")
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

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNSSplit(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(3.)*cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 18


@pytest.mark.skip(reason="implementation broken")
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

    loop = mdc.cuda_pairloop.PairLoopNeighbourListNSSplit(kernel=kernel,
                                                     dat_dict=kernel_map,
                                                     shell_cutoff=math.sqrt(3.)*cell_width+tol)

    state.nc.zero()

    loop.execute()
    nc = state.nc[:]
    for ix in range(state.npart_local):
        assert nc[ix] == 26



