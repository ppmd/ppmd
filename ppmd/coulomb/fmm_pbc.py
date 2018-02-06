from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


from math import log, ceil
from ppmd.coulomb.octal import *
import numpy as np
from ppmd import runtime, host, kernel, pairloop, data, access, mpi, opt
from ppmd.lib import build
import ctypes
import os
import math
import cmath
from threading import Thread
from scipy.special import lpmv, rgamma, gammaincc, lambertw
import sys



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
UINT64 = ctypes.c_uint64
UINT32 = ctypes.c_uint32
INT64 = ctypes.c_int64
INT32 = ctypes.c_int32

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
        vol = self.domain.extent[0] * self.domain.extent[1] * \
              self.domain.extent[2]

        kappa = math.sqrt(math.pi/(vol**(2./3.)))
        eps = min(10.**-8, self.eps)

        #print("COMPUTE SN: \t\tkappa", kappa, "vol", (vol**(2./3.)), "extent", self.domain.extent[:])

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
            maxt = max(int(math.ceil(rc/min_len)), 5)

            iterset = range(-1 * maxt, maxt+1)
            if len(iterset) < 4: print("Warning, small real space cutoff.")

            for tx in itertools.product(iterset, iterset, iterset):
                dx = tx[0]*bx + tx[1]*by + tx[2]*bz

                dispt = self._cart_to_sph(dx)

                #if dispt[0] <= rc and nd1:

                if (tx[0] != 0) or (tx[1] != 0) or (tx[2] != 0):

                    iradius = 1./dispt[0]
                    radius_coeff = iradius ** (lx + 1.)

                    kappa2radius2 = kappa2 * dispt[0] * dispt[0]

                    #mval = list(range(0, lx+1, 2))
                    mval = list(range(-1*lx, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]
                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    for mxi, mx in enumerate(mval):

                        assert abs(scipy_p[mxi].imag) < 10.**-16

                        val = math.sqrt(float(math.factorial(
                            lx - abs(mx)))/math.factorial(lx + abs(mx)))

                        ynm = val * scipy_p[mxi].real * np.cos(mx * dispt[1])

                        coeff = ynm * radius_coeff * \
                                gammaincc(lx + 0.5, kappa2radius2)

                        #print("ynm", ynm, "radius_coeff", radius_coeff, "coeff", coeff)
                        #print("lx+0.5", lx+0.5, "k2r2", kappa2radius2, "gammaincc", gammaincc(lx + 0.5, kappa2radius2))

                        terms[self.re_lm(lx, mx)] += coeff

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


        #for lx in range(2, self.L*2, 2):
        #    for mx in range(0, lx+1, 2):
        #        print("lx", lx, "mx", mx, terms[self.re_lm(lx, mx)])
        #print("G END ============================================")
        return terms

    def compute_f(self):
        #print("F START ============================================")

        ncomp = ((self.L * 2)**2) * 2
        terms = np.zeros(ncomp, dtype=self.dtype)

        extent = self.domain.extent
        lx = (extent[0], 0., 0.)
        ly = (0., extent[1], 0.)
        lz = (0., 0., extent[2])
        ivolume = 1./(extent[0]*extent[1]*extent[2])

        #gx = np.cross(ly,lz)*ivolume #* 2. * math.pi
        #gy = np.cross(lz,lx)*ivolume #* 2. * math.pi
        #gz = np.cross(lx,ly)*ivolume #* 2. * math.pi

        gx = np.array((1./extent[0], 0., 0.))
        gy = np.array((0., 1./extent[1], 0.))
        gz = np.array((0., 0., 1./extent[2]))

        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)

        for lx in range(2, self.L*2, 2):

            rc, vc, kappa = self._compute_parameters(lx)

            #print(lx, rc, vc, kappa)

            kappa2 = kappa * kappa
            mpi2okappa2 = -1.0 * (math.pi ** 2.) / kappa2

            ll = 5
            if int(ceil(vc/gxl)) < ll:
                vc = gxl*ll


            nmax_x = int(ceil(vc/gxl))
            nmax_y = int(ceil(vc/gyl))
            nmax_z = int(ceil(vc/gzl))

            #print("nmax_x", nmax_x, gx, vc)
            #print(range(-1*nmax_z, nmax_z+1))

            for hxi in itertools.product(range(-1*nmax_z, nmax_z+1),
                                          range(-1*nmax_y, nmax_y+1),
                                          range(-1*nmax_x, nmax_x+1)):

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
                    #print("lx", lx, "\thxi", hxi, "\thx", hx, "\tcoeff", coeff)

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
                #print("lx", lx, "mx", mx, terms[self.re_lm(lx, mx)])

        #print("F END ============================================")
        return terms

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

        print("\n")
        print(30*"-", "shell terms", 30*'-')
        print("radius:", limit)
        for nx in range(2, nl, 2):
            for mx in list(range(0, nx+1, 2)):
                print("nx:", nx, "\tmx:", mx,
                      "\tshell val:", terms[self.re_lm(nx, mx)],
                      "\tewald val:", self._boundary_terms[self.re_lm(nx, mx)]
                )

        print(30*"-", "-----------", 30*'-')
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


