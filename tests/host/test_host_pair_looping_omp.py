#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import math


import ppmd as md
from ppmd.access import *

N = 1000
crN = 10 #cubert(N)
E = 8.

Eo2 = E/2.

tol = 10.**(-12)


rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
GlobalArray = md.data.GlobalArray
State = md.state.State
PairLoop = md.pairloop.PairLoopNeighbourListNSOMP
Kernel = md.kernel.Kernel


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
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = GlobalArray(ncomp=1)
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
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = GlobalArray(ncomp=1)
    A.u.halo_aware = True

    return A

@pytest.fixture(scope="module", params=(0, nproc-1))
def base_rank(request):
    return request.param


def test_host_pair_loop_NS_1(state):
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
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(cell_width-tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 0



def test_host_pair_loop_NS_2(state):
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
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6


def test_host_pair_loop_NS_3(state):
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
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(math.sqrt(2.)*cell_width-tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=math.sqrt(2.)*cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6


def test_host_pair_loop_NS_4(state):
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
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(math.sqrt(2.)*cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=math.sqrt(2.)*cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 18, "ix={}".format(ix)




def test_host_pair_loop_NS_5(state):
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
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(math.sqrt(3.)*cell_width-tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=math.sqrt(3.)*cell_width-tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 18



def test_host_pair_loop_NS_6(state):
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
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(math.sqrt(3.)*cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_NS_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=math.sqrt(3.)*cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 26



def test_host_pair_loop_NS_FCC2():
    """
    Set a cutoff slightly larger than the 3rd nearest neighbour distance in
    the grid
    """
    A = State()

    crN2 = 10

    A.npart = (crN2**3)*4

    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    A.P = PositionDat(ncomp=3)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)


    cell_width = float(E)/float(crN2)

    A.P[:] = md.utility.lattice.fcc((crN2, crN2, crN2), (E, E, E))

    A.npart_local = (crN2**3)*4
    A.filter_on_domain_boundary()


    kernel_code = '''
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(cell_width-tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_NS_1',code=kernel_code)
    kernel_map = {'P': A.P(md.access.R),
                  'NC': A.nc(md.access.INC0)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=cell_width-tol)

    A.nc.zero()

    loop.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 12


def test_host_pair_loop_NS_FCC():
    """
    Set a cutoff slightly larger than the 3rd nearest neighbour distance in
    the grid
    """
    A = State()

    crN2 = 10

    A.npart = (crN2**3)*4

    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    A.P = PositionDat(ncomp=3)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)


    cell_width = (0.5*float(E))/float(crN2)

    A.P[:] = md.utility.lattice.fcc((crN2, crN2, crN2), (E, E, E))

    A.npart_local = (crN2**3)*4
    A.filter_on_domain_boundary()


    kernel_code = '''
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
    }
    ''' % {'CUTOFF': str(math.sqrt(2.)*cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_NS_1', code=kernel_code)
    kernel_map = {'P': A.P(md.access.R),
                  'NC': A.nc(md.access.INC0)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=math.sqrt(2.)*cell_width+tol)

    A.nc.zero()

    loop.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 12


def test_host_pair_loop_NS_FCC_2():
    """
    Set a cutoff slightly larger than the 3rd nearest neighbour distance in
    the grid
    """
    A = State()

    crN2 = 10

    A.npart = (crN2**3)*4

    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    A.P = PositionDat(ncomp=3)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.nn = GlobalArray(ncomp=1, dtype=ctypes.c_int)
    A.nn2 = GlobalArray(ncomp=2, dtype=ctypes.c_int)
    A.nset = GlobalArray(ncomp=1, dtype=ctypes.c_int)
    A.nset.set(4)

    cell_width = (0.5*float(E))/float(crN2)

    A.P[:] = md.utility.lattice.fcc((crN2, crN2, crN2), (E, E, E))

    A.npart_local = (crN2**3)*4
    NTOTAL = (crN2**3)*4
    A.filter_on_domain_boundary()

    kernel_code = '''
    NC.i[0]+=1;
    NN[0]++;
    NN2[0]+=NSET[0];
    NN2[1]+=2;
    '''

    kernel = md.kernel.Kernel('test_host_pair_loop_NS_1',code=kernel_code)
    kernel_map = {'P': A.P(md.access.R),
                  'NC': A.nc(md.access.INC_ZERO),
                  'NN': A.nn(md.access.INC_ZERO),
                  'NSET': A.nset(md.access.READ),
                  'NN2': A.nn2(md.access.INC_ZERO)}

    loop = PairLoop(
        kernel=kernel,
        dat_dict=kernel_map,
        shell_cutoff=math.sqrt(2.)*cell_width+tol
    )

    loop.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 12

    assert A.nn[0] == 12*NTOTAL
    assert A.nn2[0] == 12*NTOTAL*4
    assert A.nn2[1] == 12*NTOTAL*2

    loop2 = PairLoop(
        kernel=kernel,
        dat_dict=kernel_map,
        shell_cutoff=math.sqrt(2.)*cell_width-tol
    )
    loop3 = PairLoop(
        kernel=kernel,
        dat_dict=kernel_map,
        shell_cutoff=2*cell_width+tol
    )

    loop2.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 0


    loop3.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 18, '{}'.format(ix)

    loop.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 12, '{}'.format(ix)

    loop3.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 18, '{}'.format(ix)




@pytest.fixture(scope="module", params=(ctypes.c_int, ctypes.c_double))
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

    rng = np.random.RandomState(seed=2352413423)
    A1i = rng.uniform(low=10, high=20, size=1)
    PRi = rng.uniform(low=10, high=20, size=(N,1))
    PRWi = rng.uniform(low=10, high=20, size=(N,1))
    SRi = rng.uniform(low=10, high=20, size=1)
    PI0i = rng.uniform(low=10, high=20, size=1)
    PIi = rng.uniform(low=10, high=20, size=1)

    A.PR[:] = PRi[0]
    A.PRW[:] = PRWi[:]
    A.PW[:] = 0
    A.PI0[:] = PI0i[0]
    A.PI[:] = PIi[0]
    A.SR[:] = SRi[:]

    A.filter_on_domain_boundary()

    kernel_code = '''
    PW.i[0] = A1;
    PRW.i[0] = A1;
    PI0.i[0] += PR.i[0];
    PI.i[0] += SR[0];
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
        'SR' :A.SR(READ)
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


    for px in range(A.npart_local):
        assert A.PR[px, 0] == cast(PRi[0, 0]), "read only data has changed"
        assert A.PW[px, 0] == a1, "bad write only particle dat"
        assert A.PRW[px, 0] == a1, "bad read/write particle dat"
        assert abs(A.PI[px, 0] - 12*A.SR[0] - cast(PIi[0]))<10.**-10, "bad increment"
        assert abs(A.PI0[px, 0] - 12*A.PR[px, 0]) < 10.**-10, "bad zero increment"



