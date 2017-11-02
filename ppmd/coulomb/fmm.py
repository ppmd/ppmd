from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from math import log, ceil
from ppmd.coulomb.octal import *
import numpy as np
from ppmd import runtime
from ppmd.lib import build
import ctypes
import os
import math
import cmath
from scipy.special import lpmv, rgamma, gammaincc, lambertw

_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

REAL = ctypes.c_double
UINT64 = ctypes.c_uint64
UINT32 = ctypes.c_uint32
INT64 = ctypes.c_int64
INT32 = ctypes.c_int32

def _numpy_ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)

def extern_numpy_ptr(arr):
    return _numpy_ptr(arr)

def _check_dtype(arr, dtype):
    if arr.dtype != dtype:
        raise RuntimeError('Bad data type. Expected: {} Found: {}.'.format(
            str(dtype), str(arr.dtype)))

class PyFMM(object):
    def __init__(self, domain, N, eps=10.**-6, shared_memory=False):

        dtype = REAL

        self.L = int(-1*log(eps,2))
        """Number of multipole expansion coefficients"""
        self.R = int(ceil(log(N, 8)))
        """Number of levels in octal tree."""
        self.dtype = dtype
        """Floating point datatype used."""
        self.domain = domain

        self.eps = eps


        ncomp = (self.L**2) * 2
        # define the octal tree and attach data to the tree.
        self.tree = OctalTree(self.R, domain.comm)
        self.tree_plain = OctalDataTree(self.tree, ncomp, 'plain', dtype)
        self.tree_halo = OctalDataTree(self.tree, ncomp, 'halo', dtype)
        self.tree_parent = OctalDataTree(self.tree, ncomp, 'parent', dtype)
        self.entry_data = EntryData(self.tree, ncomp, dtype)

        self._tcount = runtime.OMP_NUM_THREADS if runtime.OMP_NUM_THREADS is \
            not None else 1
        self._thread_allocation = np.zeros(1, dtype=INT32)

        # load contribution computation library
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleContribution.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleContribution.h') as fh:
            hpp = fh.read()
        self._contribution_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_contrib')['particle_contribution']

        # pre compute A_n^m and 1/(A_n^m)
        self._a = np.zeros(shape=(self.L*2, (self.L*4)+1), dtype=dtype)
        self._ar = np.zeros(shape=(self.L*2,(self.L*4)+1), dtype=dtype)

        for lx in range(self.L*2):
            for mx in range(-1*lx, lx+1):
                a_l_m = ((-1.) ** lx)/math.sqrt(math.factorial(lx - mx) *\
                                                math.factorial(lx+mx))
                self._a[lx, self.L*2 + mx] = a_l_m
                self._ar[lx, self.L*2 + mx] = 1.0/a_l_m

        # pre compute the powers of i
        self._ipower_mtm = np.zeros(shape=(2*self.L+1, 2*self.L+1),
                                    dtype=dtype)
        self._ipower_mtl = np.zeros(shape=(2*self.L+1, 2*self.L+1),
                                    dtype=dtype)
        self._ipower_ltl = np.zeros(shape=(2*self.L+1, 2*self.L+1),
                                    dtype=dtype)

        for kxi, kx in enumerate(range(-1*self.L, self.L+1)):
            for mxi, mx in enumerate(range(-1*self.L, self.L+1)):

                self._ipower_mtm[kxi, mxi] = \
                    ((1.j) ** (abs(kx) - abs(mx) - abs(kx - mx))).real

                self._ipower_mtl[kxi, mxi] = \
                    ((1.j) ** (abs(kx - mx) - abs(kx) - abs(mx))).real

                self._ipower_ltl[kxi, mxi] = \
                    ((1.j) ** (abs(mx) - abs(mx - kx) - abs(kx))).real


        # pre compute the coefficients needed to compute spherical harmonics.
        self._ycoeff = np.zeros(shape=(self.L*2)**2,
                                dtype=dtype)

        for nx in range(self.L*2):
            for mx in range(-1*nx, nx+1):
                self._ycoeff[self.re_lm(nx, mx)] = math.sqrt(
                    float(math.factorial(nx - abs(mx))) /
                    float(math.factorial(nx + abs(mx)))
                )

        # As we have a "uniform" octal tree the values Y_l^m(\alpha, \beta)
        # can be pre-computed for the 8 children of a parent cell. Indexed
        # lexicographically.
        pi = math.pi
        #     (1.25 * pi, 0.75 * pi),

        alpha_beta = (
            (1.25 * pi, -1./math.sqrt(3.)),
            (1.75 * pi, -1./math.sqrt(3.)),
            (0.75 * pi, -1./math.sqrt(3.)),
            (0.25 * pi, -1./math.sqrt(3.)),
            (1.25 * pi, 1./math.sqrt(3.)),
            (1.75 * pi, 1./math.sqrt(3.)),
            (0.75 * pi, 1./math.sqrt(3.)),
            (0.25 * pi, 1./math.sqrt(3.))
        )

        self._yab = np.zeros(shape=(8, ((self.L*2)**2)*2), dtype=dtype)
        for cx, child in enumerate(alpha_beta):
            for lx in range(self.L*2):
                mval = list(range(-1*lx, 1)) + list(range(1, lx+1))
                mxval = list(range(lx, -1, -1)) + list(range(1, lx+1))
                scipy_p = lpmv(mxval, lx, child[1])
                for mxi, mx in enumerate(mval):
                    val = math.sqrt(float(math.factorial(
                        lx - abs(mx)))/math.factorial(lx + abs(mx)))
                    re_exp = np.cos(mx*child[0]) * val
                    im_exp = np.sin(mx*child[0]) * val

                    assert abs(scipy_p[mxi].imag) < 10.**-16

                    self._yab[cx, self.re_lm(lx, mx)] = \
                        scipy_p[mxi].real * re_exp
                    self._yab[cx, (self.L*2)**2 + self.re_lm(lx, mx)] = \
                        scipy_p[mxi].real * im_exp

        # load multipole to multipole translation library
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateMTM.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateMTM.h') as fh:
            hpp = fh.read()
        self._translate_mtm_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_translate_mtm')['translate_mtm']

        # load multipole to local lib
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateMTL.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateMTL.h') as fh:
            hpp = fh.read()
        self._translate_mtl_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_translate_mtl')


        # --- periodic boundaries ---
        # "Precise and Efficient Ewald Summation for Periodic Fast Multipole
        # Method", Takashi Amisaki, Journal of Computational Chemistry, Vol21,
        # No 12, 1075-1087, 2000

        # pre compute the "periodic boundaries coefficients.
        # self._boundary_terms[:] += self._compute_f() + self._compute_g()
        self._boundary_terms =  self._compute_f() + self._compute_g()

        # pre-compute spherical harmonics for interaction lists.
        # P_n^m and coefficient, 7*7*7*ncomp as offsets are in -3,3
        self._interaction_p = np.zeros((7, 7, 7, (self.L * 2)**2), dtype=dtype)

        # exp(m\phi) \phi is the longitudinal angle. 
        self._interaction_e = np.zeros((7, 7, 8*self.L + 2), dtype=dtype)
        
        # compute the lengendre polynomial coefficients
        for iz, pz in enumerate(range(-3, 4)):
            for iy, py in enumerate(range(-3, 4)):
                for ix, px in enumerate(range(-3, 4)):
                    # get spherical coord of box
                    sph = self._cart_to_sph((px, py, pz))
                    
                    for lx in range(self.L*2):
                        msci_range = list(range(lx, -1, -1)) +\
                                     list(range(1, lx+1))
                        mact_range = list(range(-1*lx, 1)) +\
                                     list(range(1, lx+1))
                        scipy_p = lpmv(msci_range, lx, math.cos(sph[2]))
                        for mxi, mx in enumerate(mact_range):
                            val = math.sqrt(math.factorial(
                                lx - abs(mx))/math.factorial(lx + abs(mx)))
                            val *= scipy_p[mxi].real
                            self._interaction_p[iz, iy, ix, 
                                self.re_lm(lx, mx)] = val

        # compute the exponential part
        for iy, py in enumerate(range(-3, 4)):
            for ix, px in enumerate(range(-3, 4)):
                # get spherical coord of box
                sph = self._cart_to_sph((px, py, 0))
                for mxi, mx in enumerate(range(-2*self.L, 2*self.L+1)):
                    self._interaction_e[iy, ix, mxi] = math.cos(mx*sph[1])
                    self._interaction_e[iy, ix, (4*self.L + 1) + mxi] = \
                        math.sin(mx*sph[1])


    def re_lm(self, l,m): return (l**2) + l + m

    def im_lm(self, l,m): return (l**2) + l +  m + self.L**2


    def _compute_cube_contrib(self, positions, charges):

        ns = self.tree.entry_map.cube_side_count
        cube_side_counts = np.array((ns, ns, ns), dtype=UINT64)
        if self._thread_allocation.size < self._tcount * \
                positions.npart_local + 1:
            self._thread_allocation = np.zeros(
                int(self._tcount*positions.npart_local*1.1 + 1),dtype=INT32)
        else:
            self._thread_allocation[:self._tcount:] = 0

        _check_dtype(positions, REAL)
        _check_dtype(charges, REAL)
        _check_dtype(self.domain.extent_internal, REAL)
        _check_dtype(self.entry_data.local_offset, UINT64)
        _check_dtype(self.entry_data.local_size, UINT64)
        _check_dtype(cube_side_counts, UINT64)
        _check_dtype(self.entry_data.data, REAL)
        _check_dtype(self._thread_allocation, INT32)


        err = self._contribution_lib(
            INT64(self.L),
            UINT64(positions.npart_local),
            INT32(self._tcount),
            positions.ctypes_data,
            charges.ctypes_data,
            self.domain.extent.ctypes_data,
            _numpy_ptr(self.entry_data.local_offset),
            _numpy_ptr(self.entry_data.local_size),
            _numpy_ptr(cube_side_counts),
            _numpy_ptr(self.entry_data.data),
            _numpy_ptr(self._thread_allocation)
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))

        self.tree_halo[self.R-1][2:-2:, 2:-2:, 2:-2:, :] = \
            self.entry_data[:,:,:,:]

    def _translate_m_to_m(self, child_level):
        """
        Translate the child expansions to their parent cells
        :return:
        """
        '''
        int translate_mtm(
            const UINT32 * RESTRICT dim_parent,     // slowest to fastest
            const UINT32 * RESTRICT dim_child,      // slowest to fastest
            const REAL * RESTRICT moments_child,
            REAL * RESTRICT moments_parent,
            const REAL * RESTRICT ylm,
            const REAL * RESTRICT alm,
            const REAL * RESTRICT almr,
            const REAL radius,
            const INT64 nlevel
        )
        '''
        _check_dtype(self.tree[child_level].parent_local_size, UINT32)
        _check_dtype(self.tree[child_level].grid_cube_size, UINT32)
        _check_dtype(self.tree_halo[child_level], REAL)
        _check_dtype(self.tree_parent[child_level], REAL)
        _check_dtype(self._yab, REAL)
        _check_dtype(self._a, REAL)
        _check_dtype(self._ar, REAL)
        _check_dtype(self._ipower_mtm, REAL)

        radius = (self.domain.extent[0] /
                 self.tree[child_level].ncubes_side_global) * 0.5

        radius = math.sqrt(radius*radius*3)

        err = self._translate_mtm_lib(
            _numpy_ptr(self.tree[child_level].parent_local_size),
            _numpy_ptr(self.tree[child_level].grid_cube_size),
            _numpy_ptr(self.tree_halo[child_level]),
            _numpy_ptr(self.tree_parent[child_level]),
            _numpy_ptr(self._yab),
            _numpy_ptr(self._a),
            _numpy_ptr(self._ar),
            _numpy_ptr(self._ipower_mtm),
            ctypes.c_double(radius),
            ctypes.c_int64(self.L)
        )

        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))

    def _fine_to_course(self, src_level):
        if src_level < 1:
            raise RuntimeError('cannot copy from a level lower than 1')

        send_parent_to_halo(src_level, self.tree_parent, self.tree_halo)

    def _image_to_sph(self, ind):
        """Convert tuple ind to spherical coordindates of periodic image."""

        dx = ind[0] * self.domain.extent[0]
        dy = ind[1] * self.domain.extent[1]
        dz = ind[2] * self.domain.extent[2]

        return self._cart_to_sph((dx, dy, dz))

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

        if lx == 2:
            return math.sqrt(3. * math.log(2. * kappa) - log(self.eps)), kappa
        else:
            n = float(lx)
            return self._compute_sn(2)
            return math.sqrt(0.5 * (2. - n) * lambertw(
                    (2./(2. - n)) * \
                    (
                        (self.eps/( (2*kappa) ** (n + 1.) ))**(2./(n - 2.))
                     )
                ).real), kappa


    def _compute_parameters(self, lx):
        if lx < 2: raise RuntimeError('not valid for lx<2')
        sn, kappa = self._compute_sn(lx)
        # r_c, v_c, kappa
        return sn/kappa, kappa*sn/math.pi, kappa


    def _compute_g(self):
        ncomp = (self.L**2) * 2
        terms = np.zeros(ncomp, dtype=self.dtype)

        min_len = min(
            self.domain.extent[0],
            self.domain.extent[1],
            self.domain.extent[2]
        )

        for lx in range(2, self.L, 2):
            rc, vc, kappa = self._compute_parameters(lx)
            kappa2 = kappa*kappa
            maxt = int(math.ceil(rc/min_len))
            iterset = range(-1 * maxt, maxt+1)
            # if len(iterset) < 4: print("Warning, small real space cutoff.")

            #print(lx, rc, vc, kappa)

            for tx in itertools.product(iterset, iterset, iterset):
                dispt = self._image_to_sph(tx)

                nd1 = abs(tx[0]) > 1 or abs(tx[1]) > 1 or abs(tx[2]) > 1

                if dispt[0] <= rc and nd1:
                    #print(tx, dispt)

                    iradius = 1./dispt[0]
                    kappa2radius2 = kappa2 * dispt[0] * dispt[0]

                    mval = list(range(-1*lx, 1, 2)) + list(range(2, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]

                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    radius_coeff = iradius ** (lx + 1.)

                    for mxi, mx in enumerate(mval):
                        val = math.sqrt(float(math.factorial(
                            lx - abs(mx)))/math.factorial(lx + abs(mx)))
                        re_exp = np.cos(mx * dispt[1]) * val
                        im_exp = np.sin(mx * dispt[1]) * val

                        assert abs(scipy_p[mxi].imag) < 10.**-16

                        coeff = scipy_p[mxi].real * radius_coeff * \
                                gammaincc(lx + 0.5, kappa2radius2)
                        #print(lx, mx, scipy_p[mxi].real * radius_coeff * \
                        #        gammaincc(lx + 0.5, kappa2radius2))

                        terms[self.re_lm(lx, mx)] += coeff * re_exp
                        terms[self.im_lm(lx, mx)] += coeff * im_exp



        return terms

    def _compute_f(self):
        ncomp = (self.L**2) * 2
        terms = np.zeros(ncomp, dtype=self.dtype)

        extent = self.domain.extent
        lx = (extent[0], 0., 0.)
        ly = (0., extent[1], 0.)
        lz = (0., 0., extent[2])
        ivolume = 1./np.dot(lx, np.cross(ly, lz))

        gx = np.cross(ly,lz)*ivolume #* 2. * math.pi
        gy = np.cross(lz,lx)*ivolume #* 2. * math.pi
        gz = np.cross(lx,ly)*ivolume #* 2. * math.pi

        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)

        #print(gx, self.domain.extent[0], gyl , gzl)

        for lx in range(2, self.L, 2):

            rc, vc, kappa = self._compute_parameters(lx)

            vc *= 4
            kappa *= 10

            #print(lx, rc, vc, kappa)

            kappa2 = kappa * kappa
            mpi2okappa2 = -1.0 * (math.pi ** 2.) / kappa2

            nmax_x = int(ceil(vc/gxl))
            nmax_y = int(ceil(vc/gyl))
            nmax_z = int(ceil(vc/gzl))

            igamma = rgamma(lx + 0.5)

            for hxi in itertools.product(range(-1*nmax_z, nmax_z+1),
                                          range(-1*nmax_y, nmax_y+1),
                                          range(-1*nmax_x, nmax_x+1)):

                hx = hxi[0]*gz + hxi[1]*gy + hxi[2]*gx
                dispt = self._cart_to_sph(hx)


                if 10.**-10 < dispt[0] <= vc:


                    exp_coeff = cmath.exp(mpi2okappa2 * dispt[0] * dispt[0]) * \
                                ivolume * igamma


                    mval = list(range(-1*lx, 1, 2)) + list(range(2, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]

                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    vhnm2 = (dispt[0] ** (lx - 2.)) * ((0 + 1.j) ** lx) * \
                            (math.pi ** (lx - 0.5))

                    #print(hx, dispt[0], "\t", abs(vhnm2*exp_coeff))

                    for mxi, mx in enumerate(mval):
                        val = math.sqrt(float(math.factorial(
                            lx - abs(mx)))/math.factorial(lx + abs(mx)))
                        re_exp = np.cos(mx * dispt[1]) * val
                        im_exp = np.sin(mx * dispt[1]) * val

                        assert abs(scipy_p[mxi].imag) < 10.**-16
                        sph_nm = re_exp * scipy_p[mxi].real + \
                                 im_exp * scipy_p[mxi].real * 1.j

                        contrib = vhnm2 * sph_nm * exp_coeff

                        terms[self.re_lm(lx, mx)] += contrib.real
                        terms[self.im_lm(lx, mx)] += contrib.imag

        return terms










