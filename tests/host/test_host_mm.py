import numpy as np
from ppmd import *

from ppmd.coulomb import mm
from ppmd.coulomb.fmm import *

import math

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double

import ppmd
from ppmd.lib import build

import time

from mpi4py import MPI

MPISIZE = MPI.COMM_WORLD.size
MPIRANK = MPI.COMM_WORLD.rank


class FreeSpaceDirect:
    def __init__(self):
        
        
        header = r"""
        #include <math.h>
        #define INT64 int64_t
        #define REAL double
        """

        src = r"""
        
        extern "C" int free_space_direct(
            const INT64 N,
            const REAL * RESTRICT P,
            const REAL * RESTRICT Q,
            REAL * RESTRICT phi
        ){{

            REAL tmp_phi = 0.0;

            #pragma omp parallel for reduction(+:tmp_phi)
            for(INT64 ix=0 ; ix<N ; ix++){{
                REAL tmp_inner_phi = 0.0;
                
                const REAL iq = Q[ix];
                const REAL ip0 = P[3*ix + 0];
                const REAL ip1 = P[3*ix + 1];
                const REAL ip2 = P[3*ix + 2];

                
                #pragma omp simd reduction(+:tmp_inner_phi)
                for(INT64 jx=(ix+1) ; jx<N ; jx++){{
                    
                    const REAL jq = Q[jx];
                    const REAL jp0 = P[3*jx + 0];
                    const REAL jp1 = P[3*jx + 1];
                    const REAL jp2 = P[3*jx + 2];

                    const REAL d0 = ip0 - jp0;
                    const REAL d1 = ip1 - jp1;
                    const REAL d2 = ip2 - jp2;
                    
                    const REAL r2 = d0*d0 + d1*d1 + d2*d2;
                    const REAL r = sqrt(r2);

                    tmp_inner_phi += iq * jq / r;

                }}
                
                tmp_phi += tmp_inner_phi;

            }}
           
            phi[0] = tmp_phi;
            return 0;
        }}
        """.format()


        self._lib = build.simple_lib_creator(header_code=header, src_code=src, name="kmc_fmm_free_space_direct")['free_space_direct']


    def __call__(self, N, P, Q):

        phi = ctypes.c_double(0)

        self._lib(
            INT64(N),
            P.ctypes.get_as_parameter(),
            Q.ctypes.get_as_parameter(),
            ctypes.byref(phi)
        )
        
        return phi.value


def test_free_space_1():

    N = 100000
    e = 10.
    R = 5
    L = 5


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)
    qi[:] = 1.0

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi,
                A.Q: qi,
            })

    
    MM = mm.PyMM(A.P, A.Q, A.domain, 'free_space', R, L)
    
    t0c = time.time()
    energy_to_test = MM(A.P, A.Q)
    t1c = time.time()
    
    t0f = time.time()
    fmm = PyFMM(A.domain, r=R, l=L, free_space=True)
    t1f = time.time()
    

    energy_fmm = fmm(A.P, A.Q)




    DFS = FreeSpaceDirect()
    


    if MPISIZE > 1:
        correct = energy_fmm
    else:
        t0 = time.time()
        correct = DFS(N, A.P.view, A.Q.view)
        t1 = time.time()

    err = abs(energy_to_test - correct) / abs(correct)
    err_fmm = abs(energy_fmm - correct) / abs(correct)
    assert err < 10.**-6
    

    #if MPIRANK == 0:
    #    print(err, err_fmm, energy_to_test, energy_fmm, correct)
    #    opt.print_profile()
    #    print(t1 - t0, t1c - t0c, t1f - t0f)








