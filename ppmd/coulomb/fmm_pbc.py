from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"



import numpy as np
from ppmd import runtime
from ppmd.lib import build
from ppmd.coulomb.octal import shell_iterator
from ppmd.coulomb.sph_harm import SphGen, LocalExpEval

import ctypes
import os
import math
import cmath
from scipy.special import lpmv, rgamma, gammaincc, lambertw
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

from math import factorial, sqrt, ceil

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

def spherical(xyz):
    """
    Converts the cartesian coordinates in xyz to spherical coordinates
    (radius, polar angle, longitude angle)
    """
    sph = np.zeros(3)
    xy = xyz[0]**2 + xyz[1]**2
    # r
    sph[0] = np.sqrt(xy + xyz[2]**2)
    # polar angle
    sph[1] = np.arctan2(np.sqrt(xy), xyz[2])
    # longitude angle
    sph[2] = np.arctan2(xyz[1], xyz[0])

    return sph


def _A(n,m):
    return ((-1.0)**n)/sqrt(factorial(n-m)*factorial(n+m))


def _h(j,k,n,m):
    if abs(k) > j: return 0.0
    if abs(m) > n: return 0.0
    if abs(m-k) > j+n: return 0.0
    icoeff = ((1.j)**(abs(k-m) - abs(k) - abs(m))).real
    return icoeff * _A(n, m) * _A(j, k) / (((-1.0) ** n) * _A(j+n, m-k))


class LongRangeMTL:
    def __init__(self, L, domain):
        self.L = L
        self.domain = domain

        _pbc_tool = FMMPbc(self.L, 10.**-10, domain, REAL)
        _rvec = _pbc_tool.compute_f() + _pbc_tool.compute_g()

        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2
        self.rmat = np.zeros((self.half_ncomp, self.half_ncomp), dtype=REAL)
        #self.rmat = csr_matrix((self.half_ncomp, self.half_ncomp), dtype=REAL)

        def _re_lm(l, m): return l**2 + l + m

        row = 0
        for jx in range(L):
            for kx in range(-jx, jx+1):
                col = 0
                for nx in range(L):
                    for mx in range(-nx, nx+1):
                        if (not abs(mx-kx) > jx+nx) and \
                            (not (abs(mx-kx) % 2 == 1)) and \
                            (not (abs(jx+nx) % 2 == 1)):
                            
                            self.rmat[row, col] = _h(jx, kx, nx, mx) * _rvec[_re_lm(jx+nx, mx-kx)]
                        col += 1
                row += 1

        self.sparse_rmat = csr_matrix(self.rmat)
        self.linop = aslinearoperator(self.rmat)
        self.sparse_linop = aslinearoperator(self.sparse_rmat)
        self.linop_data = np.array(self.sparse_rmat.data, dtype=REAL)
        self.linop_indptr = np.array(self.sparse_rmat.indptr, dtype=INT64)
        self.linop_indices = np.array(self.sparse_rmat.indices, dtype=INT64)

        # dipole correction vars
        self._dpc = DipoleCorrector(L, self.domain.extent, self._apply_operator)
        self.dipole_correction = self._dpc.scales

    def _apply_operator(self, M, L):
        L[:self.half_ncomp] = self.sparse_linop.dot(M[:self.half_ncomp])
        L[self.half_ncomp:] = self.sparse_linop.dot(M[self.half_ncomp:])

    def apply_lr_mtl(self, M, L):
        self._apply_operator(M, L)
        self._dpc(M, L)

    def __call__(self, M, L):
        return self.apply_lr_mtl(M, L)


class DipoleCorrector:
    def __init__(self, l, extent, lr_func):
        self.l = l
        self._imo = l * l
        self.ncomp = self._imo * 2
        self.extent = extent
        self.lr_func = lr_func
        self._lee = LocalExpEval(l)
        self.scales = [0,0,0]

        # x direction
        M = np.zeros(self.ncomp, dtype=REAL)
        M[self._re(1, -1)] = 1.0
        M[self._re(1,  1)] = 1.0
        eval_point = (-0.5*extent[0], 0.0, 0.0)
        lphi_sr =  self._sr_phi(M ,eval_point)
        lphi_lr= self._lr_phi(M, eval_point)
        lphi = lphi_sr + lphi_lr
        self.scales[0] = lphi

        # y direction
        M = np.zeros(self.ncomp, dtype=REAL)
        M[self._im(1, -1)] = -1.0
        M[self._im(1,  1)] =  1.0
        eval_point = (0.0, -0.5*extent[1], 0.0)
        lphi_sr =  self._sr_phi(M ,eval_point)
        lphi_lr= self._lr_phi(M, eval_point)
        lphi = lphi_sr + lphi_lr
        self.scales[1] = lphi
        
        # z direction
        M = np.zeros(self.ncomp, dtype=REAL)
        M[self._re(1, 0)] =  1.0
        eval_point = (0.0, 0.0, -0.5*extent[2])
        lphi_sr =  self._sr_phi(M ,eval_point)
        lphi_lr= self._lr_phi(M, eval_point)
        lphi = lphi_sr + lphi_lr
        self.scales[2] = lphi

        # scale x,y, and z to be the actual coefficients

        self.scales[0] *= (-1.0 * ( 2.0 ** 0.5 ) ) / (self.extent[0])
        self.scales[1] *= (-1.0 * ( 2.0 ** 0.5 ) ) / (self.extent[1])
        self.scales[2] *= 1.0 / (0.5 * self.extent[2])

    def __call__(self, M, L):

        if L is not None:
            # x direction
            xcoeff = M[self._re(1, 1)] * self.scales[0]
            #print("xcoeff", xcoeff)
            L[self._re(1, -1)] += xcoeff.real
            L[self._re(1,  1)] += xcoeff.real
            
            # y direction
            ycoeff = M[self._im(1, 1)] * self.scales[1]
            L[self._im(1, -1)] += ycoeff.real
            L[self._im(1,  1)] -= ycoeff.real
            
            # z direction
            L[self._re(1, 0)] +=  M[self._re(1, 0)] * self.scales[2]


    def _sr_phi(self, M, eval_point):
        iterset = (-1, 0, 1)
        tphi = 0.0
        re = self._re
        im = self._im
        Y = self._Y_1
        for ofx in itertools.product(iterset, iterset, iterset):
            px = [ev - ex * ox for ev, ex, ox in zip(eval_point, self.extent, ofx)]
            radius, theta, phi = spherical(px)
            ir2 = 1.0 / (radius * radius)
            tphi += (M[re(1, -1)] + M[im(1, -1)]*1.j) * Y(-1, theta, phi) * ir2
            tphi += (M[re(1,  0)] + M[im(1,  0)]*1.j) * Y( 0, theta, phi) * ir2
            tphi += (M[re(1,  1)] + M[im(1,  1)]*1.j) * Y( 1, theta, phi) * ir2

        return tphi.real
    

    def _lr_phi(self, M, eval_point):
        L = np.zeros_like(M)
        self.lr_func(M, L)
        return self._lee(L, spherical(eval_point))

    
    @staticmethod
    def _Y_1(k, theta, phi):
        if k == 0:
            return cmath.cos(theta)
        if abs(k) == 1:
            return -1.0 * cmath.sqrt(0.5) * cmath.sin(theta) * cmath.exp(k * 1.j * phi)
        else:
            raise RuntimeError('bad k value')
    
    def _re(self, l, m): return (l**2) + l + m
    def _im(self, l, m): return (l**2) + l + m + self._imo
    def _im2(self, l, m): return (l**2) + l + m + self._imo2
    @staticmethod
    def _h1k1m(k,m): 
        def _A(n,m): return ((-1.0)**n) / cmath.sqrt(factorial(n - m) * factorial(n + m))
        return ((-1.j)**(abs(k - m) - abs(k) - abs(m))) * _A(1, m) * _A(1, k) / _A(2, m - k)
    

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

        vol = self.domain.extent[0] * self.domain.extent[1] * \
              self.domain.extent[2]

        self.kappa = math.sqrt(math.pi/(vol**(2./3.)))


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

        eps = min(10.**-8, self.eps)
        kappa = self.kappa

        if lx < 2:
            raise RuntimeError('s_n cannot be computed for lx < 2.')

        elif lx == 2:
            s2 =  3. * math.log(2. * kappa) - math.log(eps)
            if s2 < 0:
                #s = 1.0
                raise RuntimeError('could not deduce (s_n)^2 (negative?)')
            else:
                s = math.sqrt(s2)
                #err = abs(s**(lx-2.) * math.exp(-1. * (s**2.)) -
                #((2.*kappa)**(-1*lx - 1.))*eps)
                #assert err<10.**-14, "LAMBERT CHECK:{}".format(err)
            return s

        else:
            n = float(lx)

            lambert_inner = (2./(2. - n)) * \
            (
                (eps/( (2*kappa) ** (n + 1.) ))**(2./(n - 2.))
            )
            tmp = 0.5 * (2. - n) * lambertw(lambert_inner, k=-1)
            
            def check_sn(sn):
                import cmath
                return (sn ** (n - 2)) * cmath.exp(-1.0 * sn * sn)

            if tmp.real < 0:
                import cmath
                sn = cmath.sqrt(tmp)
                import ipdb; ipdb.set_trace()
                
                raise RuntimeError('bad (s_n)^2 computed')
            s = math.sqrt(tmp.real)
            return s


    def _compute_parameters(self, lx):
        if lx < 2: raise RuntimeError('not valid for lx<2')
        # sn = self._compute_sn(lx)
        sn = self._compute_sn(2)
        # r_c, v_c
        return sn/self.kappa, self.kappa*sn/math.pi


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
            rc, vc = self._compute_parameters(lx)
            
            kappa = self.kappa
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
        
        
        # explicitly extract the "full" contribution from nearby cells
        # i.e. the case where the real space part included the nearest
        # neighbours
        for lx in range(2, self.L*2, 2):

            iterset = tuple(range(-1, 2, 1))

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

            rc, vc = self._compute_parameters(lx)
            kappa = self.kappa

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


