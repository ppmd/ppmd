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




@cuda
@pytest.fixture
def state(request):
    if mdc.CUDA_IMPORT_ERROR is not None:
        print mdc.CUDA_IMPORT_ERROR

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


@pytest.fixture(
    scope="module",
    params=(
            (1. , 6, 1.),
            (-1., 0, 1.),
            (1. , 18, math.sqrt(2.)),
            (-1., 6, math.sqrt(2.)),
            (1. , 26, math.sqrt(3.)),
            (-1., 18, math.sqrt(3.))
    )
)
def tolset(request):
    return request.param

@pytest.fixture(
    scope="module",
    params=(
            (0.25,),
            (0.5,),
            (0.75,),
            (1.,),
            (1.5,),
            (2.,)
    )
)
def sub_cell_factor(request):
    return request.param


#@cuda
def test_cuda_pair_loop_2(state, tolset, sub_cell_factor):
    """
    Set a cutoff slightly smaller than the smallest distance in the grid
    """

    cell_width = float(E)/float(crN)

    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in xrange(crN):
        for iy in xrange(crN):
            for iz in xrange(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()

    RC = cell_width*tolset[2] + tol*tolset[0]

    kernel_code = '''
    const double rx = P.j[0] - P.i[0];
    const double ry = P.j[1] - P.i[1];
    const double rz = P.j[2] - P.i[2];
    const double r2 = rx*rx + ry*ry + rz*rz;
    
    if (_i == 1) {
    printf("_j %%d, r2 %%f, CX %%d\\n", _j, r2, _CX);
    }
    
    
     
    NC.i[0] += (r2 < %(TOL)s) ? 1 : 0 ;
    ''' % {'TOL': str(RC**2.)}

    kernel = md.kernel.Kernel('test_cuda_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = mdc.cuda_pairloop.PairLoopCellByCell(
        kernel=kernel,
        dat_dict=kernel_map,
        shell_cutoff=RC,
        sub_divide=RC*sub_cell_factor[0]
    )

    state.nc.zero()

    loop.execute()

    print state.p[:10:,:]
    print state.nc[:N:,0]

    for ix in range(state.npart_local):
        assert state.nc[ix] == tolset[1], "ix={}".format(ix)




