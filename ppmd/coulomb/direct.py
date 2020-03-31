
import ctypes
import ppmd
from ppmd.lib import build

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from itertools import product

from ppmd.coulomb.fmm_pbc import LongRangeMTL
from ppmd.coulomb.fmm_interaction_lists import compute_interaction_lists

import numpy as np

from cgen import *

from ppmd.coulomb.sph_harm import MultipoleDotVecCreator

from collections.abc import Iterable

def spherical(xyz):
    """
    Converts the cartesian coordinates in xyz to spherical coordinates
    (radius, polar angle, longitude angle)
    
    :arg xyz: Input xyz coordinates as Numpy array or tuple/list.
    """
    if type(xyz) is tuple:
        sph = np.zeros(3)
        xy = xyz[0]**2 + xyz[1]**2
        # r
        sph[0] = np.sqrt(xy + xyz[2]**2)
        # polar angle
        sph[1] = np.arctan2(np.sqrt(xy), xyz[2])
        # longitude angle
        sph[2] = np.arctan2(xyz[1], xyz[0])

    else:
        sph = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        # r
        sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
        # polar angle
        sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
        # longitude angle
        sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

    return sph


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


class NearestDirect:
    def __init__(self, E, tuples=None):

        if not isinstance(E, Iterable):
            E = (E, E, E)
        
        
        if tuples is None:
            ox_range = tuple(range(-1, 2))
            tuples = product(ox_range, ox_range, ox_range)

        inner = ''

        for oxi, ox in enumerate(tuples):
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
                        OX=ox[0] * E[0],
                        OY=ox[1] * E[1],
                        OZ=ox[2] * E[2]
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


class FarFieldDirect:
    def __init__(self, domain, L, ex=None):

        self.lrc = LongRangeMTL(L, domain, exclude_tuples=ex)

        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2

        self._exp_compute = MultipoleDotVecCreator(L)

        self.multipole_exp = np.zeros(self.ncomp, dtype=REAL)
        self.local_dot_coeffs = np.zeros(self.ncomp, dtype=REAL)

    def __call__(self, N, P, Q):

        self.multipole_exp.fill(0)
        self.local_dot_coeffs.fill(0)
 
        for px in range(N):
            # multipole expansion for the whole cell
            self._exp_compute.dot_vec_multipole(
                spherical(tuple(P[px,:])),
                Q[px, 0],
                self.local_dot_coeffs,
                self.multipole_exp
            )


        L_tmp = np.zeros_like(self.local_dot_coeffs)
        self.lrc(self.multipole_exp, L_tmp)

        lr = 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

        return lr


class PBCDirect:
    def __init__(self, E, domain, L):

        if not isinstance(E, Iterable):
            E = (E, E, E)

        if abs(E[0] - E[1]) < 10**-14 and abs(E[0] - E[2]) < 10**-14:
            il = None
            ex = None
        else:
            il, ex = compute_interaction_lists(domain.extent)

        self._ffd = FarFieldDirect(domain, L, ex)
        self._nd = NearestDirect(E, tuples=ex)


    def __call__(self, N, P, Q):

        sr = self._nd(N, P, Q)
        lr = self._ffd(N, P, Q)

        return sr + lr



