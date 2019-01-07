from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


from math import log, ceil

from ppmd.coulomb.cached import cached

import numpy as np
from ppmd import runtime, host, kernel, pairloop, data, access, mpi, opt
from ppmd.lib import build
from ppmd.coulomb.octal import shell_iterator
from ppmd.coulomb.sph_harm import SphGen

import ctypes
import os
import math
import cmath
from threading import Thread
from scipy.special import lpmv, rgamma, gammaincc, lambertw
import sys

import itertools

def red(input):
    try:
        from termcolor import colored
        return colored(input, 'red')
    except Exception as e: return input
def green(input):
    try:
        from termcolor import colored
        return colored(input, 'green')
    except Exception as e: return input
def yellow(input):
    try:
        from termcolor import colored
        return colored(input, 'yellow')
    except Exception as e: return input

_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

REAL = ctypes.c_double
INT64 = ctypes.c_int64

np.set_printoptions(threshold=np.nan)


class FMMPbc(object):
    """
    "Precise and Efficient Ewald Summation for Periodic Fast Multipole
    Method", Takashi Amisaki, Journal of Computational Chemistry, Vol21,
    No 12, 1075-1087, 2000
    """
    def __init__(self, L, eps, domain, dtype):
        self.L = L
        self.eps = eps
        self.domain = domain
        self.dtype = dtype

        with open(str(_SRC_DIR) + \
                          '/FMMSource/PBCSource.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/PBCSource.h') as fh:
            hpp = fh.read()

        self._lib = build.simple_lib_creator(hpp, cpp,
            'pbc_setup_lib')

        self._lib['test1']()


    def re_lm(self, l,m): return (l**2) + l + m
    def im_lm(self, l,m): return (l**2) + l +  m + self.L**2


    @staticmethod
    def _cart_to_sph(xyz):
        dx = xyz[0]; dy = xyz[1]; dz = xyz[2]

        dx2dy2 = dx*dx + dy*dy
        radius = math.sqrt(dx2dy2 + dz*dz)
        phi = math.atan2(dy, dx)
        theta = math.atan2(math.sqrt(dx2dy2), dz)

        return radius, phi, theta


    def _compute_sn(self, lx):
        """
        This method tries to use the paper's parameter selection. It will
        probably fail for lx != 2. We take the kappa values for lx=2
        for lx !=2 and take increasingly large shells in real and reciprocal
        space until the values converge.
        """
        vol = self.domain.extent[0] * self.domain.extent[1] * \
              self.domain.extent[2]

        kappa = math.sqrt(math.pi/(vol**(2./3.)))
        eps = min(10.**-8, self.eps)


        if lx == 2:
            tmp = 3. * math.log(2. * kappa)
            logtmp = log(eps)
            if logtmp > tmp:
                s = 1.
                #print("BODGE WARNING")
            else:
                s = math.sqrt(3. * math.log(2. * kappa) - log(eps))
                err = abs(s**(lx-2.) * math.exp(-1. * (s**2.)) -
                ((2.*kappa)**(-1*lx - 1.))*eps)
                assert err<10.**-14, "LAMBERT CHECK:{}".format(err)

            return s, kappa
        else:
            n = float(lx)
            tmp = 0.5 * (2. - n) * lambertw(
                    (2./(2. - n)) * \
                    (
                        (eps/( (2*kappa) ** (n + 1.) ))**(2./(n - 2.))
                     )
                ).real

            #print("ARG", 0.5 * (2. - n) * lambertw(
            #        (2./(2. - n)) * \
            #        (
            #            (eps/( (2*kappa) ** (n + 1.) ))**(2./(n - 2.))
            #         )
            #    ))

            if tmp >= 0.0:
                s = math.sqrt(tmp)
                err = abs(s**(lx-2.) * math.exp(-1. * (s**2.)) -
                ((2.*kappa)**(-1*lx - 1.))*eps)
                #assert err<10.**-14, "LAMBERT CHECK: {}".format(err)
            else:
                #print("BODGE WARNING")
                s = self._compute_sn(2)[0]

            return s, kappa


    def _compute_parameters(self, lx):
        if lx < 2: raise RuntimeError('not valid for lx<2')
        sn, kappa = self._compute_sn(lx)
        sn, kappa = self._compute_sn(2)
        # r_c, v_c, kappa
        return sn/kappa, kappa*sn/math.pi, kappa


    def _image_to_sph(self, ind):
       """Convert tuple ind to spherical coordindates of periodic image."""
       dx = ind[0] * self.domain.extent[0]
       dy = ind[1] * self.domain.extent[1]
       dz = ind[2] * self.domain.extent[2]

       return self._cart_to_sph((dx, dy, dz))

    #@cached(maxsize=4096)
    def _hfoo(self, lx, mx):
        return math.sqrt(float(math.factorial(
                lx - abs(mx)))/math.factorial(lx + abs(mx)))

    def compute_g(self):

        #print("G START ============================================")

        ncomp = ((self.L * 2)**2) * 2

        terms = np.zeros(ncomp, dtype=self.dtype)

        extent = self.domain.extent
        min_len = min(extent[0], extent[1], extent[2])
        bx = np.array((extent[0], 0.0, 0.0))
        by = np.array((0.0, extent[1], 0.0))
        bz = np.array((0.0, 0.0, extent[2]))


        for lx in range(2, self.L*2, 2):
            rc, vc, kappa = self._compute_parameters(lx)

            kappa2 = kappa*kappa

            maxt = 3
            iterset = range(-1 * maxt, maxt+1)

            for tx in itertools.product(iterset, iterset, iterset):

                if (tx[0] != 0) or (tx[1] != 0) or (tx[2] != 0):

                    dx = tx[0]*bx + tx[1]*by + tx[2]*bz
                    dispt = self._cart_to_sph(dx)

                    iradius = 1./dispt[0]
                    radius_coeff = iradius ** (lx + 1.)

                    kappa2radius2 = kappa2 * dispt[0] * dispt[0]

                    #mval = list(range(0, lx+1, 2))
                    mval = list(range(-1*lx, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    for mxi, mx in enumerate(mval):

                        assert abs(scipy_p[mxi].imag) < 10.**-16

                        val = self._hfoo(lx, mx)

                        ynm = val * scipy_p[mxi].real * np.cos(mx * dispt[1])

                        coeff = ynm * radius_coeff * \
                                gammaincc(lx + 0.5, kappa2radius2)

                        terms[self.re_lm(lx, mx)] += coeff

            # add increasingly large outer shells until the values converge
            for shellx in range(maxt, 20):
                curr = np.copy(terms[self.re_lm(lx,-lx):self.re_lm(lx,lx+1):])

                for tx in shell_iterator(shellx):
                    dx = tx[0]*bx + tx[1]*by + tx[2]*bz
                    dispt = self._cart_to_sph(dx)

                    iradius = 1./dispt[0]
                    radius_coeff = iradius ** (lx + 1.)

                    kappa2radius2 = kappa2 * dispt[0] * dispt[0]

                    #mval = list(range(0, lx+1, 2))
                    mval = list(range(-1*lx, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    for mxi, mx in enumerate(mval):

                        assert abs(scipy_p[mxi].imag) < 10.**-16

                        val = self._hfoo(lx, mx)

                        ynm = val * scipy_p[mxi].real * np.cos(mx * dispt[1])

                        coeff = ynm * radius_coeff * \
                                gammaincc(lx + 0.5, kappa2radius2)

                        terms[self.re_lm(lx, mx)] += coeff

                new_vals = terms[self.re_lm(lx,-lx):self.re_lm(lx,lx+1):]
                err = np.linalg.norm(curr - new_vals, np.inf)

                if err < self.eps*0.01:
                    # print("g shellx", shellx, 10.**-15)
                    break
                if shellx == 20:
                    raise RuntimeError('Periodic Boundary Coefficients did'
                                       'not converge, Please file a bug'
                                       'report.')

        # explicitly extract the nearby cells
        for lx in range(2, self.L*2, 2):

            maxs = 1
            iterset = list(range(-1*maxs, maxs+1, 1))

            for tx in itertools.product(iterset, iterset, iterset):
                if (tx[0] != 0) or (tx[1] != 0) or (tx[2] != 0):

                    dx = tx[0]*bx + tx[1]*by + tx[2]*bz

                    dispt = self._cart_to_sph(dx)
                    iradius = 1./dispt[0]
                    radius_coeff = iradius ** (lx + 1.)

                    #mval = list(range(0, lx+1, 2))
                    mval = list(range(-1*lx, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    for mxi, mx in enumerate(mval):
                        assert abs(scipy_p[mxi].imag) < 10.**-16
                        val = math.sqrt(float(math.factorial(
                            lx - abs(mx)))/math.factorial(lx + abs(mx)))

                        ynm = val * scipy_p[mxi].real * np.cos(mx * dispt[1])
                        coeff = ynm * radius_coeff
                        terms[self.re_lm(lx, mx)] -= coeff

        return terms

    def compute_f(self):

        ncomp = ((self.L * 2)**2) * 2
        terms = np.zeros(ncomp, dtype=self.dtype)

        extent = self.domain.extent
        ivolume = 1./(extent[0]*extent[1]*extent[2])

        gx = np.array((1./extent[0], 0., 0.))
        gy = np.array((0., 1./extent[1], 0.))
        gz = np.array((0., 0., 1./extent[2]))

        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)

        for lx in range(2, self.L*2, 2):

            rc, vc, kappa = self._compute_parameters(lx)

            kappa2 = kappa * kappa
            mpi2okappa2 = -1.0 * (math.pi ** 2.) / kappa2

            ll = 6
            if int(ceil(vc/gxl)) < ll:
                vc = gxl*ll

            nmax = int(ceil(vc/gxl))
            #nmax = 1

            for hxi in itertools.product(range(- 1*nmax, nmax+1),
                                          range(-1*nmax, nmax+1),
                                          range(-1*nmax, nmax+1)):

                hx = hxi[0]*gz + hxi[1]*gy + hxi[2]*gx
                dispt = self._cart_to_sph(hx)

                if 10.**-10 < dispt[0] <= vc:

                    exp_coeff = math.exp(mpi2okappa2 * dispt[0] * dispt[0])

                    #mval = list(range(0, lx+1, 2))
                    mval = list(range(-1*lx, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    vhnm2 = ((dispt[0] ** (lx - 2.)) * ((0 + 1.j) ** lx) * \
                            (math.pi ** (lx - 0.5))).real

                    coeff = vhnm2 * exp_coeff

                    for mxi, mx in enumerate(mval):

                        val = math.sqrt(float(math.factorial(
                            lx - abs(mx)))/math.factorial(lx + abs(mx)))
                        re_exp = np.cos(mx * dispt[1]) * val

                        assert abs(scipy_p[mxi].imag) < 10.**-16
                        sph_nm = re_exp * scipy_p[mxi].real

                        contrib = sph_nm * coeff
                        terms[self.re_lm(lx, mx)] += contrib.real


        for lx in range(2, self.L*2, 2):
            igamma = rgamma(lx + 0.5) * ivolume
            for mx in range(-1*lx, lx+1, 2):
                terms[self.re_lm(lx, mx)] *= igamma

        return terms




class _shell_test_2_FMMPbc(object):
    """
    "Precise and Efficient Ewald Summation for Periodic Fast Multipole
    Method", Takashi Amisaki, Journal of Computational Chemistry, Vol21,
    No 12, 1075-1087, 2000
    """
    def __init__(self, L, eps, domain, dtype):
        self.L = L
        self.eps = eps
        self.domain = domain
        self.dtype = dtype



    def re_lm(self, l,m): return (l**2) + l + m
    def im_lm(self, l,m): return (l**2) + l +  m + self.L**2


    @staticmethod
    def _cart_to_sph(xyz):
        dx = xyz[0]; dy = xyz[1]; dz = xyz[2]

        dx2dy2 = dx*dx + dy*dy
        radius = math.sqrt(dx2dy2 + dz*dz)
        phi = math.atan2(dy, dx)
        theta = math.atan2(math.sqrt(dx2dy2), dz)

        return radius, phi, theta


    def _test_shell_sum(self, limit, nl=8):
        ncomp = ((self.L * 2)**2) * 2
        terms = np.zeros(ncomp, dtype=self.dtype)
        im_terms = np.zeros(ncomp, dtype=self.dtype)
        extent = self.domain.extent

        iterset = range(-1*limit, limit+1)
        for itx in itertools.product(iterset, iterset, iterset):
            nd1 = abs(itx[0]) > 1 or abs(itx[1]) > 1 or abs(itx[2]) > 1

            lenofvec = itx[0]**2 + itx[1]**2 + itx[2]**2
            nd2 = lenofvec < (limit**2)

            if nd1 and nd2:
                vec = np.array((itx[0]*extent[0], itx[1]*extent[1],
                                itx[2]*extent[2]))
                sph_vec = self._cart_to_sph(vec)
                ir = 1./sph_vec[0]
                for nx in range(2, nl, 2):
                    irp = ir ** (nx + 1.)
                    #mval = list(range(0, nx+1, 2))
                    mval = list(range(-1*nx, nx+1, 2))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, nx, math.cos(sph_vec[2]))
                    for mxi, mx in enumerate(mval):
                        val = math.sqrt(float(math.factorial(
                            nx - abs(mx)))/math.factorial(nx + abs(mx)))

                        re_exp =  np.cos(mx * sph_vec[1]) * val

                        sph_nm =  scipy_p[mxi].real * irp

                        terms[self.re_lm(nx, mx)] += sph_nm * re_exp

                        im_exp =  np.sin(mx * sph_vec[1]) * val
                        im_terms[self.re_lm(nx, mx)] += sph_nm * im_exp

        return terms

    def _test_shell_sum2(self, limit, nl=8):
        ncomp = ((self.L * 2)**2) * 2
        terms = np.zeros(ncomp, dtype=self.dtype)
        im_terms = np.zeros(ncomp, dtype=self.dtype)
        extent = self.domain.extent

        iterset = range(-1*limit, limit+1)
        for itx in itertools.product(iterset, iterset, iterset):
            nd1 = abs(itx[0]) > 1 or abs(itx[1]) > 1 or abs(itx[2]) > 1

            lenofvec = itx[0]**2 + itx[1]**2 + itx[2]**2
            nd2 = lenofvec < (limit**2)

            if nd1 and nd2:
                vec = np.array((itx[0]*extent[0], itx[1]*extent[1],
                                itx[2]*extent[2]))
                sph_vec = self._cart_to_sph(vec)
                ir = 1./sph_vec[0]
                for nx in range(1, nl, 1):
                    irp = ir ** (nx + 1.)
                    #mval = list(range(0, nx+1, 2))
                    mval = list(range(-1*nx, nx+1, 1))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, nx, math.cos(sph_vec[2]))
                    for mxi, mx in enumerate(mval):
                        val = math.sqrt(float(math.factorial(
                            nx - abs(mx)))/math.factorial(nx + abs(mx)))

                        re_exp =  np.cos(mx * sph_vec[1]) * val

                        sph_nm =  scipy_p[mxi].real * irp

                        terms[self.re_lm(nx, mx)] += sph_nm * re_exp

                        im_exp =  np.sin(mx * sph_vec[1]) * val
                        im_terms[self.re_lm(nx, mx)] += sph_nm * im_exp


        return terms


class SphShellSum(object):
    def __init__(self, lmax):
        self._lmax = lmax
        im_of = (lmax+1) ** 2
        self._ncomp = 2 * im_of
        self.ncomp = self._ncomp

        sph_gen = SphGen(lmax)
        
        def lm_ind(L, M, OX=0):
            return ((L) * ( (L) + 1 ) + (M) + OX)

        radius_gen = 'const double iradius = 1.0/radius;\nconst double r0 = 1.0;\n'
        assign_gen = ''
        for lx in range(lmax+1):
            radius_gen += 'const double r{lxp1} = r{lx} * iradius;\n'.format(lxp1=lx+1, lx=lx)

            for mx in range(-lx, lx+1):
                assign_gen += 'tmp_out[{LM_IND}] += '.format(LM_IND=lm_ind(lx, mx)) + \
                    str(sph_gen.get_y_sym(lx, mx)[0]) + \
                    ' * r{lx};\n'.format(lx=lx+1)
                assign_gen += 'tmp_out[{LM_IND}] += '.format(LM_IND=lm_ind(lx, mx, im_of)) + \
                    str(sph_gen.get_y_sym(lx, mx)[1]) + \
                    ' * r{lx};\n'.format(lx=lx+1)


        src = """
        #include <omp.h>
        #define STRIDE ({STRIDE})

        extern "C" int sph_gen(
            const int num_threads,
            const int N,
            const double * RESTRICT radius_set,
            const double * RESTRICT theta_set,
            const double * RESTRICT phi_set,
            double * RESTRICT gtmp_out,
            double * RESTRICT out
        ){{
            for(int tx=0 ; tx<(num_threads*STRIDE) ; tx++){{
                gtmp_out[tx] = 0;
            }}
            omp_set_num_threads(num_threads);

            #pragma omp parallel default(none) shared(N, radius_set, theta_set, phi_set, gtmp_out)
            {{

                const int threadid = omp_get_thread_num();
                const int inner_num_threads = omp_get_num_threads();

                const int lower = N*threadid/inner_num_threads;
                const int upper = (threadid == (inner_num_threads - 1)) ? N : N*(threadid+1)/inner_num_threads;
                
                double * RESTRICT tmp_out = gtmp_out + threadid * STRIDE;
                
                for (int ix=lower; ix<upper ; ix++){{
                    const double radius = radius_set[ix];
                    const double theta = theta_set[ix];
                    const double phi = phi_set[ix];
                    {RADIUS_GEN}
                    {SPH_GEN}
                    {ASSIGN_GEN}
                }}

            }}

            for(int tx=0 ; tx<num_threads ; tx++){{
                for(int ix=0 ; ix<STRIDE ; ix++){{
                    out[ix] += gtmp_out[ix + tx*STRIDE];
                }}
            }}

            return 0;
        }}
        """.format(
            RADIUS_GEN=radius_gen,
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            STRIDE=self._ncomp
        )
        header = str(sph_gen.header)

        self._lib = build.simple_lib_creator(header_code=header, src_code=src)['sph_gen']
        self._nthreads = runtime.NUM_THREADS
        self._gtmp = np.zeros(self._ncomp*self._nthreads, dtype=ctypes.c_double)

    def __call__(self, radius, theta, phi, out):
        
        assert radius.dtype == ctypes.c_double
        assert theta.dtype == ctypes.c_double
        assert phi.dtype == ctypes.c_double
        assert out.dtype == ctypes.c_double
        assert len(theta) == len(phi)
        assert len(theta) == len(radius)
        assert len(out) == self._ncomp

        N = len(theta)

        self._lib(
            ctypes.c_int(self._nthreads),
            ctypes.c_int(N),
            radius.ctypes.get_as_parameter(),
            theta.ctypes.get_as_parameter(),
            phi.ctypes.get_as_parameter(),
            self._gtmp.ctypes.get_as_parameter(),
            out.ctypes.get_as_parameter()
        )


