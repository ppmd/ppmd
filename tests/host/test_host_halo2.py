#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md

N = 1000
crN = 10 #cubert(N)
E = 8.

Eo2 = E/2.
tol = 0.1


rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


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

one_proc = pytest.mark.skipif("md.mpi.MPI.COMM_WORLD.Get_size() > 1")

@one_proc
def test_host_halo_1_cell(state):
    """
    Check cell counts before and after halo exchange.
    """
    cell_width = float(E)
    state.domain.cell_decompose(E)
    state.get_cell_to_particle_map().create()
    state.get_cell_to_particle_map().update_required = True


    pi = md.utility.lattice.cubic_lattice((crN,crN,crN), (E, E, E))

    state.p[:] = pi
    state.npart_local = N

    state.filter_on_domain_boundary()


    state.get_cell_to_particle_map().check()

    ca = state.domain.cell_array


    for cx in range(ca[0]):
        for cy in range(ca[1]):
            for cz in range(ca[2]):

                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                if ( cx == 0 or cx == ca[0]-1 ) or \
                   ( cy == 0 or cy == ca[1]-1 ) or \
                   ( cz == 0 or cz == ca[2]-1 ):
                    assert state.get_cell_to_particle_map().cell_contents_count[ci] == 0
                else:
                    assert state.get_cell_to_particle_map().cell_contents_count[ci] == N

    state.p.halo_exchange()

    for cx in range(ca[0]):
        for cy in range(ca[1]):
            for cz in range(ca[2]):
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                assert state.get_cell_to_particle_map().cell_contents_count[ci] == N







