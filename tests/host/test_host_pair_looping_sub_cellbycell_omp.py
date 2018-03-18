#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import math

np.set_printoptions(threshold=np.nan)

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
PairLoop = md.pairloop.SubCellByCellOMP
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
    A.f4 = ParticleDat(ncomp=4)
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

@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param


def test_host_pair_loop_NS_1(state):
    """
    Set a cutoff slightly smaller than the smallest distance in the grid
    """
    cell_width = float(E)/float(crN)
    pi = np.zeros([N,3], dtype=ctypes.c_double)

    pi = md.utility.lattice.cubic_lattice((crN, crN, crN), (E,E,E))

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
    Tests a global array
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
    
    ga = GlobalArray(size=2, dtype=ctypes.c_int64)


    kernel_code = '''
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
        GA[0] += 1;
    }
    GA[1]+=1;
    ''' % {'CUTOFF': str(cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W),
                  'GA': ga(md.access.INC)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6

    assert ga[0] == 6*N


def test_host_pair_loop_NS_4(state):
    """
    Tests a scalar array read
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
    
    ga = GlobalArray(size=2, dtype=ctypes.c_int64)
    sa = ScalarArray(ncomp=3, dtype=ctypes.c_double)
    sa[0] = math.pi
    sa[0] = math.pi*0.5
    sa[0] = 1000.*math.pi

    kernel_code = '''
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    if ((r0*r0 + r1*r1 + r2*r2) <= %(CUTOFF)s*%(CUTOFF)s){
        NC.i[0]+=1;
        GA[0] += 1;
        F.i[0] += SA[0];
        F.i[1] += 2.*SA[1];
        F.i[2] += 3.*SA[2];
    }
    GA[1]+=1;
    ''' % {'CUTOFF': str(cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {'P': state.p(md.access.R),
                  'NC': state.nc(md.access.W),
                  'F': state.f(md.access.W),
                  'GA': ga(md.access.INC),
                  'SA': sa(md.access.READ)}

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=cell_width+tol)

    state.nc.zero()

    loop.execute()
    for ix in range(state.npart_local):
        assert state.nc[ix] == 6
    
    for ix in range(state.npart_local):
        assert abs(state.f[ix, 0] - 6*sa[0]) < 10.**-14
        assert abs(state.f[ix, 1] - 12*sa[1]) < 10.**-14
        assert abs(state.f[ix, 2] - 18*sa[2]) < 10.**-14


    assert ga[0] == 6*N


def test_host_pair_loop_NS_5(state):
    """
    Tests a scalar array read
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
    
    ga = GlobalArray(size=2, dtype=ctypes.c_double)
    sa = ScalarArray(ncomp=3, dtype=ctypes.c_double)
    sa[0] = math.pi
    sa[0] = math.pi*0.5
    sa[0] = 1000.*math.pi

    kernel_code = '''
    #define rc2 1.0
    #define CF 1.0
    #define sigma2 1.0
    #define CV 1.0
    #define internalshift 1.0
    const double R0 = P.j[0] - P.i[0];
    const double R1 = P.j[1] - P.i[1];
    const double R2 = P.j[2] - P.i[2];

    const double r2 = R0*R0 + R1*R1 + R2*R2;

    const double r_m2 = sigma2/r2;
    const double r_m4 = r_m2*r_m2;
    const double r_m6 = r_m4*r_m2;

    const double r_m8 = r_m4*r_m4;
    const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

    F.i[0]+= (r2 < rc2) ? f_tmp*R0 : 0.0;
    F.i[1]+= (r2 < rc2) ? f_tmp*R1 : 0.0;
    F.i[2]+= (r2 < rc2) ? f_tmp*R2 : 0.0;
    u[0]+= (r2 < rc2) ? 0.5*CV*((r_m6-1.0)*r_m6 + internalshift) : 0.0;

    ''' % {'CUTOFF': str(cell_width+tol)}

    kernel = md.kernel.Kernel('test_host_pair_loop_1',code=kernel_code)
    kernel_map = {
            'P': state.p(md.access.R),
            'F': state.f(md.access.W),
            'u': ga(md.access.INC_ZERO)
            }

    loop = PairLoop(kernel=kernel,
                    dat_dict=kernel_map,
                    shell_cutoff=cell_width+tol)

    state.nc.zero()

    loop.execute()
   
