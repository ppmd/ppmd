#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import math

np.set_printoptions(threshold=np.nan)

import ppmd as md
from ppmd.access import *

def red(*input):
    try:
        from termcolor import colored
        return colored(*input, color='red')
    except Exception as e: return input
def green(*input):
    try:
        from termcolor import colored
        return colored(*input, color='green')
    except Exception as e: return input
def yellow(*input):
    try:
        from termcolor import colored
        return colored(*input, color='yellow')
    except Exception as e: return input



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

PairLoopNL = md.pairloop.PairLoopNeighbourListNSOMP

def red_tol(val, tol):
    if abs(val) > tol:
        return red(str(val))
    else:
        return green(str(val))


def test_host_pair_loop_NS_5():
    """
    Tests a scalar array read
    """
    
    crN = 50 #cubert(N)
    N = crN**3
    E = 3. * crN

    Eo2 = E/2.   


    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.f2 = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    cell_width = 6.0 * float(E)/float(crN)

    rng = np.random.RandomState(seed=865)

    A.p[:] = rng.uniform(low=-0.4999*E, high=0.4999*E, size=(N,3))
    

    A.npart_local = N
    A.filter_on_domain_boundary()
    
    ga = GlobalArray(size=1, dtype=ctypes.c_double)
    ga2 = GlobalArray(size=1, dtype=ctypes.c_double)

    kernel_code = '''
    #define rc2 %(CUTOFF)s*%(CUTOFF)s
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

    kernel = md.kernel.Kernel('test_host_compare_1',code=kernel_code)

    loop = PairLoopNL(kernel=kernel,
                    dat_dict={
                        'P': A.p(md.access.R),
                        'F': A.f(md.access.W),
                        'u': ga(md.access.INC_ZERO)
                    },
                    shell_cutoff=cell_width+tol)
    loop2 = PairLoop(kernel=kernel,
                    dat_dict={
                        'P': A.p(md.access.R),
                        'F': A.f2(md.access.W),
                        'u': ga2(md.access.INC_ZERO)
                    },
                    shell_cutoff=cell_width+tol)


    A.nc.zero()

    loop.execute()
    loop2.execute()
    

    DEBUG = True
    assert ((ga[0] - ga2[0])/ga[0]) < 10.**-14
    for px in range(A.npart_local):
        m = np.linalg.norm(A.f[px,:])
        if m == 0.0:
            m = 1.0
        err = np.linalg.norm((A.f[px,:] - A.f2[px,:])/m, np.inf)
        assert err < 10.**-13
        if err > 10.**-10 and DEBUG:
            print(px, red_tol(err, 10.**-6), A.f[px, :], A.f2[px,:])
    
    if DEBUG:
        md.opt.print_profile()
    
    nl_time = md.opt.PROFILE['PairLoopNeighbourListNSOMP:test_host_compare_1:execute_internal']
    nl_count = md.opt.PROFILE['PairLoopNeighbourListNSOMP:test_host_compare_1:kernel_execution_count']

    flop_count = 27
    nl_rate = (nl_count*flop_count/nl_time)/(10.**9)

    c_time = md.opt.PROFILE['SubCellByCellOMP:test_host_compare_1:execute_internal']
    c_count = md.opt.PROFILE['SubCellByCellOMP:test_host_compare_1:kernel_execution_count']

    c_rate = (c_count*flop_count/c_time)/(10.**9)
    
    print("nlist rate:\t", nl_rate)
    print("cell  rate:\t", c_rate)
