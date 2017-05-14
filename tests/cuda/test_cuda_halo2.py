#!/usr/bin/python

import pytest
import ctypes
import numpy as np



import ppmd as md
import ppmd.cuda as mdc


cuda = pytest.mark.skipif("mdc.CUDA_IMPORT is False")

N = 8000
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

def bool_in_halo(x,y,z,cx,cy,cz):
    if x==0 or y==0 or z==0:
        return 1
    elif x==cx-1 or y==cy-1 or z==cz-1:
        return 1
    if x==1 or y==1 or z==1:
        return 0
    elif x==cx-2 or y==cy-2 or z==cz-2:
        return 0
    else:
        return -1


@cuda
def test_cuda_halo_cell_check(state):
    """
    Check cell counts before and after halo exchange.
    """
    C0 = 7
    C1 = 6
    C2 = 5

    state.domain.cell_array[:] = np.array((C0,C1,C2))
    state.get_cell_to_particle_map()._update_cell_in_halo()

    flags = state.get_cell_to_particle_map().cell_in_halo_flag[:]

    for cz in xrange(C2):
        for cy in xrange(C1):
            for cx in xrange(C0):

                cv = cx + C0*(cy + cz*C1)
                assert bool_in_halo(cx,cy,cz,C0,C1,C2) == flags[cv]



@cuda
def test_host_halo_cube_4(state, h_state):
    """
    Check cell contents using a cell by cell inspection.
    """

    if nproc > 1:
        cell_width = float(E)/float(crN)

        state.domain.cell_decompose(cell_width)
        state.get_cell_to_particle_map().create()
        state.get_cell_to_particle_map().update_required = True

        h_state.domain.cell_decompose(cell_width)
        h_state.get_cell_to_particle_map().create()
        h_state.get_cell_to_particle_map().update_required = True


        pi = np.random.uniform(-0.5*E, 0.5*E, [N,3])
        pi = np.array(pi, dtype=ctypes.c_double)


        state.p[:] = pi
        state.npart_local = N
        state.filter_on_domain_boundary()
        state.get_cell_to_particle_map().check()


        h_state.p[:] = pi
        h_state.npart_local = N
        h_state.filter_on_domain_boundary()
        h_state.get_cell_to_particle_map().check()

        state.p.halo_exchange()
        h_state.p.halo_exchange()


        assert h_state.p.npart_local == state.p.npart_local
        assert h_state.p.npart_local_halo == state.p.npart_local_halo

        cl = h_state.p.npart_local
        ch = h_state.p.npart_local_halo


        h_p = h_state.p[cl:cl+ch:, :]
        d_p = state.p[cl:cl+ch:, :]


        h_p = h_p[np.lexsort((h_p[:, 0], h_p[:, 1], h_p[:,2]))]
        d_p = d_p[np.lexsort((d_p[:, 0], d_p[:, 1], d_p[:,2]))]

        '''
        print rank, h_p.shape, d_p.shape, ch

        np.set_printoptions(linewidth=60)

        if rank == 0:
            print rank, "DEV local"
            print state.p[0:cl:,::]

            print rank, "DEV halo"
            print state.p[cl:cl+ch:,::]

            print rank, "HOST halo"
            print h_state.p[cl:cl+ch:,::]

        '''

        for ix in range(ch):
            assert h_p[ix, 0] == d_p[ix, 0]
            assert h_p[ix, 1] == d_p[ix, 1]
            assert h_p[ix, 2] == d_p[ix, 2]






