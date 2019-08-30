import numpy as np

import pytest
from ppmd import *

from ppmd.coulomb import mm
from ppmd.coulomb import lm
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

from itertools import product


from ppmd.coulomb.fmm_pbc import LongRangeMTL

class NearestDirect:
    def __init__(self, E):

        ox_range = tuple(range(-1, 2))

        inner = ''

        for oxi, ox in enumerate(product(ox_range, ox_range, ox_range)):
                if ox[0] != 0 or ox[1] != 0 or ox[2] != 0:
                    inner += """
                            d0 = jp0 - ip0 + {OX};
                            d1 = jp1 - ip1 + {OY};
                            d2 = jp2 - ip2 + {OZ};
                            r2 = d0*d0 + d1*d1 + d2*d2;
                            r = sqrt(r2);
                            tmp_inner_phi += 0.5 * iq * jq / r;

                    """.format(
                        OXI=oxi,
                        OX=ox[0] * E,
                        OY=ox[1] * E,
                        OZ=ox[2] * E
                    )
        
        
        header = r"""
        #include <math.h>
        #define INT64 int64_t
        #define REAL double
        """

        src = r"""
        
        extern "C" int nearest_direct(
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

                for(INT64 jx=(ix+1) ; jx<N ; jx++){{
                    
                    const REAL jq = Q[jx];
                    const REAL jp0 = P[3*jx + 0];
                    const REAL jp1 = P[3*jx + 1];
                    const REAL jp2 = P[3*jx + 2];

                    REAL d0 = ip0 - jp0;
                    REAL d1 = ip1 - jp1;
                    REAL d2 = ip2 - jp2;
                    
                    REAL r2 = d0*d0 + d1*d1 + d2*d2;
                    REAL r = sqrt(r2);

                    tmp_inner_phi += iq * jq / r;

                }}

                for(INT64 jx=0 ; jx<N ; jx++){{
                    
                    const REAL jq = Q[jx];
                    const REAL jp0 = P[3*jx + 0];
                    const REAL jp1 = P[3*jx + 1];
                    const REAL jp2 = P[3*jx + 2];

                    REAL d0;
                    REAL d1;
                    REAL d2;
                    
                    REAL r2;
                    REAL r;

                    {INNER}

                }}
                
                tmp_phi += tmp_inner_phi;

            }}
           
            phi[0] = tmp_phi;
            return 0;
        }}
        """.format(
            INNER=inner
        )

        self._lib = build.simple_lib_creator(header_code=header, src_code=src, name="kmc_fmm_nearest_direct")['nearest_direct']


    def __call__(self, N, P, Q):

        phi = ctypes.c_double(0)

        self._lib(
            INT64(N),
            P.ctypes.get_as_parameter(),
            Q.ctypes.get_as_parameter(),
            ctypes.byref(phi)
        )
        
        return phi.value



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




@pytest.mark.parametrize("MM_LM", (mm.PyMM, lm.PyLM))
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_free_space_1(MM_LM, BC):
    

    N = 10000
    e = 10.
    R = 5
    L = 16


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)

    if BC == 'pbc':
        bias = np.sum(qi) / N
        qi -= bias
        assert abs(np.sum(qi)) < 10.**-12




    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi,
                A.Q: qi,
            })

    
    MM = MM_LM(A.P, A.Q, A.domain, BC, R, L)

    fmm_bc = {
        'free_space': True,
        '27': '27',
        'pbc': False
    }[BC]

    fmm = PyFMM(A.domain, r=R, l=L, free_space=fmm_bc)
    

    t0c = time.time()
    energy_to_test = MM(A.P, A.Q)
    t1c = time.time()
    

    t0f = time.time()
    # energy_fmm = fmm(A.P, A.Q)
    t1f = time.time()


    if BC == 'free_space':
        DFS = FreeSpaceDirect()
    elif BC == '27':
        DFS = NearestDirect(e)


    if MPISIZE > 1:
        correct = fmm(A.P, A.Q)
        t0 = 0.0
        t1 = 0.0
    else:
        if not BC == 'pbc':
            t0 = time.time()
            correct = DFS(N, A.P.view, A.Q.view)
            t1 = time.time()
        else:
            correct = fmm(A.P, A.Q)


    err = abs(energy_to_test - correct) / abs(correct)
    assert err < 10.**-6
    
    return
    if MPIRANK == 0:
        #print(err, err_fmm, energy_to_test, energy_fmm, correct)
        print("Direct", t1 - t0, MM_LM, t1c - t0c, "FMM", t1f - t0f)








