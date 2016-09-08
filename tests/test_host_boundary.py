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
    offset[:,2] = -0.5*(E - tol)*np.ones([crN*crN], dtype=ctypes.c_double)

    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,2] = E*np.ones([crN*crN], dtype=ctypes.c_double)

    pj = pi + offset
    pjc = pi - offset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)


    kernel_code = '''
    P(2) -= %(TOL)s ;
   ''' % {'TOL': str(tol)}

    kernel = md.kernel.Kernel('test_host_boundary_z0',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)


    state.domain.boundary_condition.apply()
    state.gather_data_on(0)

    if rank == 0:

        inds = state.gid[:,0].argsort()
        pp = state.p[inds]

        assert np.sum(np.abs(pp - pjc)) < 1.


def test_host_boundary_z1(state):

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
    offset[:,2] = -0.5*(E - tol)*np.ones([crN*crN], dtype=ctypes.c_double)

    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,2] = E*np.ones([crN*crN], dtype=ctypes.c_double)

    pj = pi - offset
    pjc = pi + offset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)


    kernel_code = '''
    P(2) += %(TOL)s ;
   ''' % {'TOL': str(tol)}

    kernel = md.kernel.Kernel('test_host_boundary_z1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)


    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    state.domain.boundary_condition.apply()

    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    # avoid excessive copying
    ps = state.p[:]
    b = state.domain.boundary[:]
    for px in range(state.npart_local):
        assert  b[0] < ps[px,0] < b[1]
        assert  b[2] < ps[px,1] < b[3]
        assert  b[4] < ps[px,2] < b[5]
    state.gather_data_on(0)

    if rank == 0:

        inds = state.gid[:,0].argsort()
        pp = state.p[inds]

        assert np.sum(np.abs(pp - pjc)) < 1.







def test_host_boundary_x0(state):

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
            pi[i, ::] = np.array([ 0, d2[d1x, d2x], d1[d1x, d2x] ])


    # pj sets particles to be outside the domain, they should be swapped
    # to the opposite side of the domain


    offset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    offset[:,0] = -0.5*(E - tol)*np.ones([crN*crN], dtype=ctypes.c_double)

    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,0] = E*np.ones([crN*crN], dtype=ctypes.c_double)

    pj = pi + offset
    pjc = pi - offset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)


    kernel_code = '''
    P(0) -= %(TOL)s ;
   ''' % {'TOL': str(tol)}

    kernel = md.kernel.Kernel('test_host_boundary_x0',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)


    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    state.domain.boundary_condition.apply()

    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    # avoid excessive copying
    ps = state.p[:]
    b = state.domain.boundary[:]
    for px in range(state.npart_local):
        assert  b[0] < ps[px,0] < b[1]
        assert  b[2] < ps[px,1] < b[3]
        assert  b[4] < ps[px,2] < b[5]
    state.gather_data_on(0)

    if rank == 0:

        inds = state.gid[:,0].argsort()
        pp = state.p[inds]

        assert np.sum(np.abs(pp - pjc)) < 1.


def test_host_boundary_x1(state):

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
            pi[i, ::] = np.array([ 0, d2[d1x, d2x], d1[d1x, d2x] ])


    # pj sets particles to be outside the domain, they should be swapped
    # to the opposite side of the domain


    offset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    offset[:,0] = -0.5*(E - tol)*np.ones([crN*crN], dtype=ctypes.c_double)

    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,0] = E*np.ones([crN*crN], dtype=ctypes.c_double)

    pj = pi - offset
    pjc = pi + offset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)


    kernel_code = '''
    P(0) += %(TOL)s ;
   ''' % {'TOL': str(tol)}

    kernel = md.kernel.Kernel('test_host_boundary_x1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)


    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    state.domain.boundary_condition.apply()

    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    # avoid excessive copying
    ps = state.p[:]
    b = state.domain.boundary[:]
    for px in range(state.npart_local):
        assert  b[0] < ps[px,0] < b[1]
        assert  b[2] < ps[px,1] < b[3]
        assert  b[4] < ps[px,2] < b[5]
    state.gather_data_on(0)

    if rank == 0:

        inds = state.gid[:,0].argsort()
        pp = state.p[inds]

        assert np.sum(np.abs(pp - pjc)) < 1.





def test_host_boundary_y0(state):

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
            pi[i, ::] = np.array([d2[d1x, d2x], 0, d1[d1x, d2x] ])


    # pj sets particles to be outside the domain, they should be swapped
    # to the opposite side of the domain


    offset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    offset[:,1] = -0.5*(E - tol)*np.ones([crN*crN], dtype=ctypes.c_double)

    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,1] = E*np.ones([crN*crN], dtype=ctypes.c_double)

    pj = pi + offset
    pjc = pi - offset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)


    kernel_code = '''
    P(1) -= %(TOL)s ;
   ''' % {'TOL': str(tol)}

    kernel = md.kernel.Kernel('test_host_boundary_y0',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)


    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    state.domain.boundary_condition.apply()

    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    # avoid excessive copying
    ps = state.p[:]
    b = state.domain.boundary[:]
    for px in range(state.npart_local):
        assert  b[0] < ps[px,0] < b[1]
        assert  b[2] < ps[px,1] < b[3]
        assert  b[4] < ps[px,2] < b[5]
    state.gather_data_on(0)

    if rank == 0:

        inds = state.gid[:,0].argsort()
        pp = state.p[inds]

        assert np.sum(np.abs(pp - pjc)) < 1.




def test_host_boundary_y1(state):

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
            pi[i, ::] = np.array([d2[d1x, d2x], 0, d1[d1x, d2x] ])


    # pj sets particles to be outside the domain, they should be swapped
    # to the opposite side of the domain


    offset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    offset[:,1] = -0.5*(E - tol)*np.ones([crN*crN], dtype=ctypes.c_double)

    coffset = np.zeros([crN*crN, 3], dtype=ctypes.c_double)
    coffset[:,1] = E*np.ones([crN*crN], dtype=ctypes.c_double)

    pj = pi - offset
    pjc = pi + offset

    #setup positions outside the boundary
    state.p[:] = pj
    state.v[:] = np.zeros([N,3])
    state.f[:] = np.zeros([N,3])
    state.gid[:,0] = np.arange(N)


    state.scatter_data_from(0)


    kernel_code = '''
    P(1) += %(TOL)s ;
   ''' % {'TOL': str(tol)}

    kernel = md.kernel.Kernel('test_host_boundary_y1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.RW)}

    loop = md.loop.ParticleLoop(kernel=kernel, particle_dat_dict=kernel_map)
    loop.execute(n=state.npart_local)


    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    state.domain.boundary_condition.apply()

    tcs = np.array([state.npart_local])
    tcr = np.array([0])
    md.mpi.MPI_HANDLE.comm.Allreduce(tcs, tcr)
    assert tcr[0] == N

    # avoid excessive copying
    ps = state.p[:]
    b = state.domain.boundary[:]
    for px in range(state.npart_local):
        assert  b[0] < ps[px,0] < b[1]
        assert  b[2] < ps[px,1] < b[3]
        assert  b[4] < ps[px,2] < b[5]
    state.gather_data_on(0)

    if rank == 0:

        inds = state.gid[:,0].argsort()
        pp = state.p[inds]

        assert np.sum(np.abs(pp - pjc)) < 1.




