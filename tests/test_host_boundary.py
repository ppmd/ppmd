#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md

N = 100
crN = 10 #sqrt(N)
E = 8.
Eo2 = E/2.
tol = 0.1


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






def test_host_boundary_z0(state):
    
    # crN, Number of particles per coordinate direction
    state.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    state.domain.boundary_condition.set_state(state)


    a = np.linspace(-0.5*(E-tol), 0.5*(E-tol), crN)
    grid = np.meshgrid(a,a)
    d1 = grid[0]
    d2 = grid[1]
    
    pi = np.zeros([crN*crN, 3], dtype=ctypes.c_double)

    
    for d1x in range(crN):
        for d2x in range(crN):
            i = d1x*crN + d2x
            pi[i, ::] = np.array([d1[d1x, d2x], d2[d1x, d2x], 0 ])
    
    
    # pj sets particles to be outside the domain, they should be swapped
    # to the opposite side of the domain
    
    
    offset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    offset[:,2] = 0.5*(E + tol)*np.ones([crN*crN], dtype=ctypes.c_double)
    
    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,2] = E*np.ones([crN*crN], dtype=ctypes.c_double)
    


    pj = pi - offset
    pjc = pj + coffset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)
    state.domain.boundary_condition.apply()
    state.gather_data_on(0)
    

    if rank == 0:

        inds = state.gid[:,0].argsort()
        print "inds", inds
        pp = state.p[inds]
        print "pp",pp, "pjc", pjc

        assert np.sum(np.abs(pp - pjc)) < 1.
    


    
























