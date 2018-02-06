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

from ppmd.cuda import CUDA_IMPORT
from ppmd.coulomb.wigner import Rzyz_set


import pytest

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

def _pdb_drop():
    #import pytest; pytest.set_trace()
    import ipdb; ipdb.set_trace()

def _isnormal(arr):
    return not (np.any(np.isinf(arr)) or np.any(np.isnan(arr)))

def _numpy_ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)

def extern_numpy_ptr(arr):
    return _numpy_ptr(arr)

def _check_dtype(arr, dtype):
    if arr.dtype != dtype:
        raise RuntimeError('Bad data type. Expected: {} Found: {}.'.format(
            str(dtype), str(arr.dtype)))
    if issubclass(type(arr), np.ndarray): return _numpy_ptr(arr)
    elif issubclass(type(arr), host.Matrix): return arr.ctypes_data
    elif issubclass(type(arr), host.Array): return arr.ctypes_data
    else: raise RuntimeError('unknown array type passed: {}'.format(type(arr)))

def _get_iarray(l):
    b = 'static const REAL _IARRAY[%(length)s] = {' % {'length': 2*l+1}
    for mx in range(-1*l, l):
        b += '{}, '.format((-1.0)**(mx))

    b += '%(LL)s};' % {'LL': (-1.0)**l}
    b += '\nstatic const REAL * RESTRICT IARRAY = &_IARRAY[%(I)s];' % \
         {'I': l}

    return b


class PyFMM(object):
    def __init__(self, domain, N=None, eps=10.**-6,
        free_space=False, r=None, shell_width=0.0, cuda=False, cuda_levels=2,
        force_unit=1.0, energy_unit=1.0, _debug=False, l=None):

        self._debug = _debug

        dtype = REAL

        if l is None:
            self.L = int(-1*log(eps,2))
            """Number of multipole expansion coefficients"""
        else:
            self.L = l

        if r is None: self.R = int(log(N, 8))
        else: self.R = int(r)
        """Number of levels in octal tree."""


        self.dtype = dtype
        """Floating point datatype used."""
        self.domain = domain

        self.eps = eps

        self.free_space = free_space

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
        self._tmp_cell = np.zeros(1, dtype=INT32)

        # pre compute A_n^m and 1/(A_n^m)

        ASTRIDE1 = (self.L*4)+1
        ASTRIDE2 = self.L*2

        self._a = np.zeros(shape=(ASTRIDE2, ASTRIDE1), dtype=dtype)
        self._ar = np.zeros(shape=(ASTRIDE2, ASTRIDE1), dtype=dtype)

        for lx in range(self.L*2):
            for mx in range(-1*lx, lx+1):
                a_l_m = ((-1.) ** lx)/math.sqrt(math.factorial(lx - mx) *\
                                                math.factorial(lx+mx))
                self._a[lx, self.L*2 + mx] = a_l_m
                # the array below is used directly in the precomputation of
                # Y_{j+n}^{m-k}
                self._ar[lx, self.L*2 + mx] = 1.0/a_l_m

        self._arn0 = np.zeros(self.L*2, dtype=dtype)
        for lx in range(2*self.L):
            a_l0 = ((-1.0)**(lx)) / float(math.factorial(lx))
            self._arn0[lx] = 1./a_l0


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
        
        # create array for j/k lookup incides
        self._j_array = np.zeros(self.L**2, dtype=INT32)
        self._k_array = np.zeros(self.L**2, dtype=INT32)
        self._a_inorder = np.zeros(self.L**2, dtype=dtype) 
        for jx in range(self.L):
            for kx in range(-1*jx, jx+1):
                self._j_array[self.re_lm(jx,kx)] = jx
                self._k_array[self.re_lm(jx,kx)] = kx
                self._a_inorder[self.re_lm(jx,kx)] = \
                    ((-1.) ** jx)/math.sqrt(math.factorial(jx - kx) *\
                    math.factorial(jx+kx))

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
                mxval = [abs(mx) for mx in mval]

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

        # local to local lib
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateLTL.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateLTL.h') as fh:
            hpp = fh.read()
        self._translate_ltl_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_translate_ltl')        

        # load contribution computation library
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleContribution.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleContribution.h') as fh:
            hpp = fh.read()
        self._contribution_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_contrib')['particle_contribution']

        # load extraction computation library
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleExtraction.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/ParticleExtraction.h') as fh:
            hpp = fh.read()
        self._extraction_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_extract')['particle_extraction']

        # load multipole to local lib z-direction only
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateMTLZ.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/TranslateMTLZ.h') as fh:
            hpp = fh.read()

        hpp = hpp % {
            'SUB_ASTRIDE1': ASTRIDE1,
            'SUB_ASTRIDE2': ASTRIDE2,
            'SUB_IARRAY': _get_iarray(self.L)
        }

        self._translate_mtlz_lib = build.simple_lib_creator(hpp, cpp,
            'fmm_translate_mtlz')

        # --- periodic boundaries ---
        # "Precise and Efficient Ewald Summation for Periodic Fast Multipole
        # Method", Takashi Amisaki, Journal of Computational Chemistry, Vol21,
        # No 12, 1075-1087, 2000

        # pre compute the periodic boundaries coefficients.
        if free_space == False:
            self._boundary_terms = self._compute_f() + self._compute_g()
            #self._boundary_terms = np.zeros((self.L * 2)**2, dtype=dtype)

        # create a vectors with ones for real part and zeros for imaginary part
        # to feed into MTL translation. Use radius=1.

        self._boundary_ident = np.zeros(8*self.L + 2, dtype=dtype)
        self._boundary_ident[:4*self.L+1:] = 1.0

        # --- end of periodic boundaries ---

        # pre-compute spherical harmonics for interaction lists.
        # P_n^m and coefficient, 7*7*7*ncomp as offsets are in -3,3
        self._interaction_p = np.zeros((7, 7, 7, (self.L * 2)**2), dtype=dtype)

        # exp(m\phi) \phi is the longitudinal angle. 
        self._interaction_e = np.zeros((7, 7, 8*self.L + 2), dtype=dtype)

        self._wigner_real = np.zeros((7,7,7), dtype=ctypes.c_void_p)
        self._wigner_imag = np.zeros((7,7,7), dtype=ctypes.c_void_p)

        self._wigner_b_real = np.zeros((7,7,7), dtype=ctypes.c_void_p)
        self._wigner_b_imag = np.zeros((7,7,7), dtype=ctypes.c_void_p)

        # storage to prevent matrices/pointer arrays going out of scope
        self._wigner_matrices_f = {}
        self._wigner_matrices_b = {}
        self._wigner_real_pointers = []
        self._wigner_imag_pointers = []

        # compute the lengendre polynomial coefficients
        for iz, pz in enumerate(range(-3, 4)):
            for iy, py in enumerate(range(-3, 4)):
                for ix, px in enumerate(range(-3, 4)):
                    # get spherical coord of box
                    # r, phi, theta
                    sph = self._cart_to_sph((px, py, pz))
                    
                    for lx in range(self.L*2):
                        mact_range = list(range(-1*lx, 1)) +\
                                     list(range(1, lx+1))

                        msci_range = [abs(mx) for mx in mact_range]

                        scipy_p = lpmv(msci_range, lx, math.cos(sph[2]))
                        for mxi, mx in enumerate(mact_range):
                            val = math.sqrt(float(math.factorial(
                                lx - abs(mx)))/math.factorial(lx + abs(mx)))

                            val *= scipy_p[mxi].real
                            # pre compute the 1./A_{j+n}^{m-k}
                            val *= self._ar[lx, self.L*2 + mx]

                            if abs(scipy_p[mxi].imag) > 10.**-15:
                                raise RuntimeError('unexpected imag part')
                            self._interaction_p[iz, iy, ix, 
                                self.re_lm(lx, mx)] = val

                    # sph = self._cart_to_sph((px, py, pz))
                    # forward rotate
                    pointers_real, pointers_imag, matrices = Rzyz_set(
                        p=self.L,
                        alpha=sph[1], beta=sph[2], gamma=0.0,
                        dtype=self.dtype)
                    # store the temporaries
                    self._wigner_matrices_f[(pz, py, px)] = matrices
                    self._wigner_real_pointers.append(pointers_real)
                    self._wigner_imag_pointers.append(pointers_imag)
                    # pointers
                    self._wigner_real[iz, iy, ix] = \
                        pointers_real.ctypes.data
                    self._wigner_imag[iz, iy, ix] = \
                        pointers_imag.ctypes.data

                    # backward rotate
                    pointers_real, pointers_imag, matrices = Rzyz_set(
                        p=self.L,
                        alpha=0.0, beta=-1.*sph[2], gamma=-1.*sph[1],
                        dtype=self.dtype)
                    # store the temporaries
                    self._wigner_matrices_b[(pz, py, px)] = matrices
                    self._wigner_real_pointers.append(pointers_real)
                    self._wigner_imag_pointers.append(pointers_imag)
                    # pointers
                    self._wigner_b_real[iz, iy, ix] = \
                        pointers_real.ctypes.data
                    self._wigner_b_imag[iz, iy, ix] = \
                        pointers_imag.ctypes.data


        # compute the exponential part (not needed for rotated mtl)
        for iy, py in enumerate(range(-3, 4)):
            for ix, px in enumerate(range(-3, 4)):
                # get spherical coord of box
                sph = self._cart_to_sph((px, py, 0))
                for mxi, mx in enumerate(range(-2*self.L, 2*self.L+1)):
                    self._interaction_e[iy, ix, mxi] = math.cos(mx*sph[1])
                    self._interaction_e[iy, ix, (4*self.L + 1) + mxi] = \
                        math.sin(mx*sph[1])


        # create a pairloop for finest level part
        P = data.ParticleDat(ncomp=3, dtype=dtype)
        F = data.ParticleDat(ncomp=3, dtype=dtype)
        Q = data.ParticleDat(ncomp=1, dtype=dtype)
        FMM_CELL = data.ParticleDat(ncomp=1, dtype=ctypes.c_int)
        self.particle_phi = data.GlobalArray(ncomp=1, dtype=dtype)
        ns = self.tree.entry_map.cube_side_count
        maxe = np.max(self.domain.extent[:]) / ns


        # zero the mask if interacting over a periodic boundary
        free_space_mod = """
        const int maskx = 1.0;
        const int masky = 1.0;
        const int maskz = 1.0;
        """
        if free_space == True:
            free_space_mod = """
            #define ABS(x) ((x) > 0 ? (x) : (-1*(x)))
            const int maskx = (ABS(P.j[0]) > {hex}) ? 0.0 : 1.0;
            const int masky = (ABS(P.j[1]) > {hey}) ? 0.0 : 1.0;
            const int maskz = (ABS(P.j[2]) > {hez}) ? 0.0 : 1.0;
            """.format(**{
                'hex': self.domain.extent[0] * 0.5,
                'hey': self.domain.extent[1] * 0.5,
                'hez': self.domain.extent[2] * 0.5
            })

        with open(str(_SRC_DIR) + \
                          '/FMMSource/CellByCellKernel.cpp') as fh:
            pair_kernel_src = str(fh.read())

        pair_kernel_src = pair_kernel_src.format(**{
            'nsx': ns,
            'nsy': ns,
            'nsz': ns,
            'hex': self.domain.extent[0] * 0.5,
            'hey': self.domain.extent[1] * 0.5,
            'hez': self.domain.extent[2] * 0.5,
            'lx': ns / self.domain.extent[0],
            'ly': ns / self.domain.extent[1],
            'lz': ns / self.domain.extent[2],
            'FREE_SPACE': free_space_mod,
            'ENERGY_UNIT': str(float(energy_unit)),
            'FORCE_UNIT': str(float(force_unit))
        })
        pair_kernel = kernel.Kernel('fmm_pairwise', code=pair_kernel_src, 
            headers=(kernel.Header('math.h'),))

        cell_by_cell = True
        if cell_by_cell:
            PL = pairloop.CellByCellOMP
            max_radius = 2.0 * (maxe + shell_width)
        else:
            PL = pairloop.PairLoopNeighbourListNSOMP
            max_radius = 1. * ((((maxe+shell_width)*2.)**2.)*3.)**0.5

        self._pair_loop = PL(
            kernel=pair_kernel,
            dat_dict={
                'P':P(access.READ),
                'Q':Q(access.READ),
                'F':F(access.INC),
                'FMM_CELL': FMM_CELL(access.READ),
                'PHI':self.particle_phi(access.INC_ZERO)
            },
            shell_cutoff=max_radius
        )
 
        self._int_list = list(range(self.R))
        self._int_list[0] = None
        for lvlx in range(1, self.R):
            tsize = self.tree[lvlx].grid_cube_size
            if tsize is not None:
                self._int_list[lvlx] = compute_interaction_lists(tsize)
            else: self._int_list[lvlx] = None
         
        self._int_tlookup = compute_interaction_tlookup()
        self._int_plookup = compute_interaction_plookup()
        self._int_radius = compute_interaction_radius()

        # profiling
        self.timer_contrib = opt.Timer(runtime.TIMER)
        self.timer_extract = opt.Timer(runtime.TIMER)

        self.timer_mtm = opt.Timer(runtime.TIMER)
        self.timer_mtl = opt.Timer(runtime.TIMER)
        self.timer_ltl = opt.Timer(runtime.TIMER)
        self.timer_local = opt.Timer(runtime.TIMER)

        self.timer_mtl_cuda = [opt.Timer(runtime.TIMER) for tx \
                               in range(cuda_levels)]

        self.timer_halo = opt.Timer(runtime.TIMER)
        self.timer_down = opt.Timer(runtime.TIMER)
        self.timer_up = opt.Timer(runtime.TIMER)

        self.execution_count = 0
        self._mtlz_ops = self._mtlz_cost_per_cell()

        # threading
        self._async_thread = None
        self._thread_space = host.ThreadSpace(n=ncomp*2, dtype=dtype)

        self.cuda = cuda
        self.cuda_levels = cuda_levels
        self.cuda_async_threads = []
        for tx in range(self.L):
            self.cuda_async_threads.append(None)

        self._cuda_mtl = None
        if self.cuda and CUDA_IMPORT:
            from . import fmm_cuda
            self._cuda_mtl = fmm_cuda.TranslateMTLCuda(
                dtype=self.dtype,
                tree=self.tree,
                nlevel=self.L,
                a_arr=self._a,
                ar_arr=self._ar,
                p_arr=self._interaction_p,
                e_arr=self._interaction_e,
                int_list=self._int_list,
                int_tlookup=self._int_tlookup,
                int_plookup=self._int_plookup,
                int_radius=self._int_radius,
                ipower_mtl=self._ipower_mtl,
                wigner_f=self._wigner_matrices_f,
                wigner_b=self._wigner_matrices_b,
                arn0=self._arn0
            )

        if self.cuda and (self._cuda_mtl is None):
            raise RuntimeError('CUDA support was requested but intialisation'
                               ' failed')


    def _update_opt(self):
        p = opt.PROFILE
        b = self.__class__.__name__ + ':'
        p[b+'num_levels'] = self.R
        p[b+'num_terms'] = self.L
        p[b+'contrib'] = self.timer_contrib.time()
        p[b+'extract'] = self.timer_extract.time()
        p[b+'mtm'] = self.timer_mtm.time()
        p[b+'mtl'] = self.timer_mtl.time()
        p[b+'mtl_gflops'] = self.flop_rate_mtl() / (10.**9.)
        p[b+'ltl'] = self.timer_ltl.time()
        p[b+'local'] = self.timer_local.time()
        p[b+'halo'] = self.timer_halo.time()
        p[b+'down'] = self.timer_down.time()
        p[b+'up'] = self.timer_up.time()
        p[b+'exec_count'] = self.execution_count


        if self.cuda:
            p[b+'mtl_cuda_(async)'] = self._cuda_mtl.timer_mtl.time()
            p[b+'mtl_cuda_gflops'] = self.cuda_flop_rate_mtl() / (10.**9.)
    def _compute_local_interaction(self, positions, charges, forces=None):

        if forces is None:
            forces = data.ParticleDat(ncomp=3, npart=positions.npart_total,
                                      dtype=self.dtype)

        self.timer_local.start()
        self._pair_loop.execute(
            dat_dict = {
                'P':positions(access.READ),
                'Q':charges(access.READ),
                'F':forces(access.INC),
                'FMM_CELL':positions.group._fmm_cell(access.READ),
                'PHI':self.particle_phi(access.INC_ZERO)
            }
        )
        self.timer_local.pause()
        return self.particle_phi[0]

    def re_lm(self, l,m): return (l**2) + l + m
    def im_lm(self, l,m): return (l**2) + l +  m + self.L**2

    def _check_aux_dat(self, positions):
        if not hasattr(positions.group, '_fmm_cell'):
            positions.group._fmm_cell = data.ParticleDat(ncomp=1, dtype=INT32)
            positions.group._fmm_cell.npart_local = positions.npart_local

    def _cuda_translate_m_t_l(self, level, async=True):
        if self.tree[level].local_grid_cube_size is None:
            return
        if self._cuda_mtl is None:
            raise RuntimeError("cuda mtl is None")
        cl = self.R - level - 1
        if cl >= self.cuda_levels:
            print("level", level, "max", self.cuda_levels, "cl", cl)
            raise RuntimeError("cuda mtl called higher than requested max")

        self.timer_mtl_cuda[cl].start()
        radius = self.domain.extent[0] / \
                 self.tree[level].ncubes_side_global
        if async:
            self._cuda_mtl_start_async(level, radius)
        else:
            self._cuda_mtl.translate_mtl(self.tree_halo, level, radius,
                                         self.tree_plain)

        self.timer_mtl_cuda[cl].pause()

    def _cuda_mtl_start_async(self, level, radius):
        thread = self.cuda_async_threads[level]
        if thread is not None:
            raise RuntimeError('Expected None, found a thread')

        self._cuda_mtl.translate_mtl_pre(level, self.tree_halo)

        func = self._cuda_mtl.translate_mtl_async_func

        self.cuda_async_threads[level]=Thread(target=func,args=(level, radius))
        self.cuda_async_threads[level].start()

    def _cuda_mtl_wait_async(self, level):
        thread = self.cuda_async_threads[level]
        if thread is None:
            return
        else:
            thread.join()
            self._cuda_mtl.translate_mtl_post(level, self.tree_plain)
            self.cuda_async_threads[level] = None


    def __call__(self, positions, charges, forces=None, async=False):


        self.entry_data.zero()
        self.tree_plain.zero()
        self.tree_halo.zero()
        self.tree_parent.zero()

        self._check_aux_dat(positions)

        self._compute_cube_contrib(positions, charges,
                                   positions.group._fmm_cell)


        for level in range(self.R - 1, 0, -1):

            self._level_call_async(self._translate_m_to_m, level, async)
            self._halo_exchange(level)

            if self.cuda and (self.R - level -1 < self.cuda_levels):
                self._cuda_translate_m_t_l(level)
            else:
                self._level_call_async(self._translate_m_to_l,
                                       level, async)

            self._fine_to_coarse(level)

            if level > 1:
                self.tree_parent[level][:] = 0.0


        self._join_async()

        if self._debug:
            self.up = np.copy(self.tree_parent[1][0,0,0, :],)

        self.tree_parent[0][:] = 0.0
        self.tree_plain[0][:] = 0.0

        self._compute_periodic_boundary()

        for level in range(1, self.R):

            if self.cuda:
                self._cuda_mtl_wait_async(level)
            self._translate_l_to_l(level)
            self._coarse_to_fine(level)

        phi_extract = self._compute_cube_extraction(positions, charges,
                                                    forces=forces)
        phi_near = self._compute_local_interaction(positions, charges,
                                                   forces=forces)

        self._update_opt()

        #print("Far:", phi_extract, "Near:", phi_near)
        self.execution_count += 1
        return phi_extract + phi_near

    def _level_call_async(self, func, level, async):

        # check previous call finished
        if async and self._async_thread is not None:
            self._async_thread.join()
            self._async_thread = None
        if async:
            self._async_thread = Thread(target=func, args=(level,))
            self._async_thread.start()
        else:
            func(level)

    def _mtlz_cost_per_cell(self):
        t = 0
        for mx in range(self.L):
            n = 2*(mx+1)+1
            t += (2*(n**2))*8

        t *= 2

        for jx in range(self.L):
            for nx in range(self.L):
                kmax = min(nx, jx)
                t += 2
                for kx in range(-kmax, kmax+1):
                    t += 7

        return t


    def flop_rate_mtl(self):
        start = 1
        if self.cuda:
            end = -1*self.cuda_levels
        else:
            end = None

        local_plain_cells = self.tree_plain.num_cells(start, end)

        # "cartesian"
        cost_per_cell = (self.L**4) * 16
        # z rotation
        cost_per_cell = self._mtlz_ops

        flop_count = 189 * cost_per_cell * local_plain_cells
        total_cost = flop_count * self.execution_count

        if self.timer_mtl.time() < 10.**-10:
            return 0.0

        return total_cost/self.timer_mtl.time()

    def cuda_flop_rate_mtl(self):
        if not self.cuda:
            return 0.0

        start = -1*self.cuda_levels
        end = None

        local_plain_cells = self.tree_plain.num_cells(start, end)
        # "cartesian"
        cost_per_cell = (self.L**4) * 16
        # z rotation
        cost_per_cell = self._mtlz_ops
        flop_count = 189 * cost_per_cell * local_plain_cells
        total_cost = flop_count * self.execution_count
        if self._cuda_mtl.timer_mtl.time() < 10.**-10:
            return 0.0
        return total_cost/self._cuda_mtl.timer_mtl.time()


    def _compute_periodic_boundary(self):

        lsize = self.tree[1].parent_local_size

        if self.free_space == '27' or self.free_space == True:
            if lsize is not None:
                self.tree_parent[1][:] = 0

            return

        if lsize is not None:

            moments = np.copy(self.tree_parent[1][0, 0, 0, :])

            self.tree_parent[1][0, 0, 0, :] = 0.0
            self._translate_mtl_lib['mtl_test_wrapper'](
                ctypes.c_int64(self.L),
                ctypes.c_double(1.),            #radius=1
                extern_numpy_ptr(moments),
                extern_numpy_ptr(self._boundary_ident),
                extern_numpy_ptr(self._boundary_terms),
                extern_numpy_ptr(self._a),
                extern_numpy_ptr(self._ar),
                extern_numpy_ptr(self._ipower_mtl),
                extern_numpy_ptr(self.tree_parent[1][0, 0, 0, :])
            )


    def _join_async(self):
        if self._async_thread is not None:
            self._async_thread.join()
            self._async_thread = None

    def _compute_cube_contrib(self, positions, charges, fmm_cell):

        self.timer_contrib.start()

        ns = self.tree.entry_map.cube_side_count
        cube_side_counts = np.array((ns, ns, ns), dtype=UINT64)
        if self._thread_allocation.size < self._tcount * \
                (positions.npart_local + 1):
            self._thread_allocation = np.zeros(
                int(self._tcount*(positions.npart_local*1.1 + 1)),dtype=INT32)

        self._thread_allocation[:self._tcount:] = 0

        if self._tmp_cell.shape[0] < positions.npart_local:
            self._tmp_cell = np.zeros(int(positions.npart_local*1.1),
                                      dtype=INT32)

        err = self._contribution_lib(
            INT64(self.L),
            UINT64(positions.npart_local),
            INT32(self._tcount),
            _check_dtype(positions, REAL),
            _check_dtype(charges, REAL),
            _check_dtype(self._tmp_cell, INT32),
            _check_dtype(self.domain.extent, REAL),
            _check_dtype(self.entry_data.local_offset, UINT64),
            _check_dtype(self.entry_data.local_size, UINT64),
            _check_dtype(cube_side_counts, UINT64),
            _check_dtype(self.entry_data.data, REAL),
            _check_dtype(self._thread_allocation, INT32)
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))

        self.tree_halo[self.R-1][2:-2:, 2:-2:, 2:-2:, :] = 0.0
        self.entry_data.add_onto(self.tree_halo)

        # hack to ensure halo exchange
        fmm_cell[:positions.npart_local:, 0] = \
            self._tmp_cell[:positions.npart_local:]

        self.timer_contrib.pause()

    def _compute_cube_extraction(self, positions, charges, forces=None):

        if forces is None:
            forces = data.ParticleDat(ncomp=3, npart=positions.npart_local,
                                      dtype=self.dtype)


        self.timer_extract.start()
        ns = self.tree.entry_map.cube_side_count
        cube_side_counts = np.array((ns, ns, ns), dtype=UINT64)

        phi = REAL(0)
        self.entry_data.extract_from(self.tree_plain)


        err = self._extraction_lib(
            INT64(self.L),
            UINT64(positions.npart_local),
            INT32(self._tcount),
            _check_dtype(positions, REAL),
            _check_dtype(charges, REAL),
            _check_dtype(forces, REAL),
            _check_dtype(self._tmp_cell, INT32),
            _check_dtype(self.domain.extent, REAL),
            _check_dtype(self.entry_data.local_offset, UINT64),
            _check_dtype(self.entry_data.local_size, UINT64),
            _check_dtype(cube_side_counts, UINT64),
            _check_dtype(self.entry_data.data, REAL),
            ctypes.byref(phi),
            _check_dtype(self._thread_allocation, INT32),
            _check_dtype(self._a, REAL),
            _check_dtype(self._ar, REAL),
            _check_dtype(self._ipower_ltl, REAL),
            INT32(0)
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))

        #print("far", positions.npart_local, phi.value)

        red_re = mpi.all_reduce(np.array((phi.value)))
        forces.ctypes_data_post(access.W)
        self.timer_extract.pause()
        return red_re


    def _translate_m_to_m(self, child_level):
        """
        Translate the child expansions to their parent cells
        :return:
        """

        self.timer_mtm.start()
        if self.tree[child_level].parent_local_size is None:
            return 

        self.tree_parent[child_level][:] = 0.0

        radius = (self.domain.extent[0] /
                 self.tree[child_level].ncubes_side_global) * 0.5

        radius = math.sqrt(radius*radius*3)

        err = self._translate_mtm_lib(
            _check_dtype(self.tree[child_level].parent_local_size, UINT32),
            _check_dtype(self.tree[child_level].grid_cube_size, UINT32),
            _check_dtype(self.tree_halo[child_level], REAL),
            _check_dtype(self.tree_parent[child_level], REAL),
            _check_dtype(self._yab, REAL),
            _check_dtype(self._a, REAL),
            _check_dtype(self._ar, REAL),
            _check_dtype(self._ipower_mtm, REAL),
            REAL(radius),
            INT64(self.L)
        )

        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))
        self.timer_mtm.pause()

    def _translate_l_to_l(self, child_level):
        """
        Translate parent expansion to child boxes on child_level. Takes parent
        data from the parent data tree.
        :param child_level: Level to translate on.
        """

        self.timer_ltl.start()
        if self.tree[child_level].local_grid_cube_size is None:
            return

        radius = (self.domain.extent[0] /
                 self.tree[child_level].ncubes_side_global) * 0.5

        radius = math.sqrt(radius*radius*3)
        err = self._translate_ltl_lib['translate_ltl'](
            _check_dtype(self.tree[child_level].parent_local_size, UINT32),
            _check_dtype(self.tree[child_level].local_grid_cube_size, UINT32),
            _check_dtype(self.tree_plain[child_level], REAL),
            _check_dtype(self.tree_parent[child_level], REAL),
            _check_dtype(self._yab, REAL),
            _check_dtype(self._a, REAL),
            _check_dtype(self._ar, REAL),
            _check_dtype(self._ipower_ltl, REAL),
            REAL(radius),
            INT64(self.L)
        )

        if err < 0: raise RuntimeError('negative return code: {}'.format(err))
        self.timer_ltl.pause()

    def _halo_exchange(self, level):
        self.timer_halo.start()
        self.tree_halo.halo_exchange_level(level)

        if self.tree[level].local_grid_cube_size is None:
            return

        # if computing the free space solution we need to zero the outer
        # halo regions

        if self.free_space == True:
            gs = self.tree[level].ncubes_side_global
            lo = self.tree[level].local_grid_offset
            ls = self.tree[level].local_grid_cube_size

            if lo[2] == 0:
                self.tree_halo[level][:,:,:2:,:] = 0.0
            if lo[1] == 0:
                self.tree_halo[level][:,:2:,:,:] = 0.0
            if lo[0] == 0:
                self.tree_halo[level][:2:,:,:,:] = 0.0
            if lo[2] + ls[2] == gs:
                self.tree_halo[level][:,:,-2::,:] = 0.0
            if lo[1] + ls[1] == gs:
                self.tree_halo[level][:,-2::,:,:] = 0.0
            if lo[0] + ls[0] == gs:
                self.tree_halo[level][-2::,:,:,:] = 0.0


        self.timer_halo.pause()

    def _translate_m_to_l(self, level):

        if self.tree[level].local_grid_cube_size is None:
            return

        self.timer_mtl.start()
        self.tree_plain[level][:] = 0.0

        radius = self.domain.extent[0] / \
                 self.tree[level].ncubes_side_global

        err = self._translate_mtlz_lib['translate_mtl'](
            _check_dtype(self.tree[level].local_grid_cube_size, UINT32),
            _check_dtype(self.tree_halo[level], REAL),
            _check_dtype(self.tree_plain[level], REAL),
            self._wigner_real.ctypes.get_as_parameter(),
            self._wigner_imag.ctypes.get_as_parameter(),
            self._wigner_b_real.ctypes.get_as_parameter(),
            self._wigner_b_imag.ctypes.get_as_parameter(),
            _check_dtype(self._a, REAL),
            _check_dtype(self._arn0, REAL),
            _check_dtype(self._ipower_mtl, REAL),
            REAL(radius),
            INT64(self.L),
            _check_dtype(self._int_list[level], INT32),
            _check_dtype(self._int_tlookup, INT32),
            _check_dtype(self._int_plookup, INT32),
            _check_dtype(self._int_radius, ctypes.c_double),
            self._thread_space.ctypes_data
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))
        self.timer_mtl.pause()


    def _translate_m_to_l_cart(self, level):

        if self.tree[level].local_grid_cube_size is None:
            return

        self.timer_mtl.start()
        self.tree_plain[level][:] = 0.0

        radius = self.domain.extent[0] / \
                 self.tree[level].ncubes_side_global

        err = self._translate_mtl_lib['translate_mtl'](
            _check_dtype(self.tree[level].local_grid_cube_size, UINT32),
            _check_dtype(self.tree_halo[level], REAL),
            _check_dtype(self.tree_plain[level], REAL),
            _check_dtype(self._interaction_e, REAL),
            _check_dtype(self._interaction_p, REAL),
            _check_dtype(self._a, REAL),
            _check_dtype(self._ar, REAL),
            _check_dtype(self._ipower_mtl, REAL),
            REAL(radius),
            INT64(self.L),
            _check_dtype(self._int_list[level], INT32),
            _check_dtype(self._int_tlookup, INT32),
            _check_dtype(self._int_plookup, INT32),
            _check_dtype(self._int_radius, ctypes.c_double),
            _check_dtype(self._j_array  ,INT32),
            _check_dtype(self._k_array  ,INT32),
            _check_dtype(self._a_inorder,REAL) 
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))
        self.timer_mtl.pause()


    def _fine_to_coarse(self, src_level):
        if src_level < 1:
            raise RuntimeError('cannot copy from a level lower than 1')
        elif src_level >= self.R:
            raise RuntimeError('cannot copy from a greater than {}'.format(
            self.R))

        if self.tree[src_level].parent_local_size is None:
            return
        self.timer_up.start()
        send_parent_to_halo(src_level, self.tree_parent, self.tree_halo)
        self.timer_up.pause()
    

    def _coarse_to_fine(self, src_level):
        if src_level == self.R - 1:
            return

        if src_level < 1:
            raise RuntimeError('cannot copy from a level lower than 1')
        elif src_level >= self.R-1:
            raise RuntimeError('cannot copy from a greater than {}'.format(
            self.R-2))
        
        self.timer_down.start()
        send_plain_to_parent(src_level, self.tree_plain, self.tree_parent)
        self.timer_down.pause()

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


    def _compute_g(self):

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

    def _compute_f(self):
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

        print("\n")
        print(30*"-", "shell terms", 30*'-')
        print("radius:", limit)
        for nx in range(2, nl, 2):
            for mx in list(range(0, nx+1, 2)):
                print("nx:", nx, "\tmx:", mx,
                      "\tshell val:", terms[self.re_lm(nx, mx)],
                      "\tewald val:", self._boundary_terms[self.re_lm(nx, mx)]
                )

        print(30*"-", "im    terms", 30*'-')
        print("radius:", limit)
        for nx in range(1, nl, 1):
            for mx in list(range(-1*nx, nx+1, 1)):
                ival = im_terms[self.re_lm(nx, mx)]
                if abs(ival) > 10.**-8:
                    str_val = red(ival)
                else:
                    str_val = str(ival)

                print("nx:", nx, "\tmx:", mx,
                      "\tshell val:", str_val,
                )

        return terms




