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

_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

REAL = ctypes.c_double
UINT64 = ctypes.c_uint64
UINT32 = ctypes.c_uint32
INT64 = ctypes.c_int64
INT32 = ctypes.c_int32

np.set_printoptions(threshold=np.nan)

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

class PyFMM(object):
    def __init__(self, domain, N=None, eps=10.**-6,
        free_space=False, r=None, shell_width=0.0):

        dtype = REAL

        self.L = int(-1*log(eps,2))
        """Number of multipole expansion coefficients"""
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
            'fmm_contrib')['particle_extraction']

        # --- periodic boundaries ---
        # "Precise and Efficient Ewald Summation for Periodic Fast Multipole
        # Method", Takashi Amisaki, Journal of Computational Chemistry, Vol21,
        # No 12, 1075-1087, 2000

        # pre compute the periodic boundaries coefficients.
        _false = False
        if not free_space:
            self._boundary_terms = self._compute_f() + self._compute_g()
            #self._boundary_terms = np.zeros((self.L * 2)**2, dtype=dtype)

            #print(60*"-")
            #print(self._boundary_terms)
            #print("max:", np.max(self._boundary_terms))
            #print(60*"-")


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
                            if abs(scipy_p[mxi].imag) > 10.**-15:
                                raise RuntimeError('unexpected imag part')
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


        # create a pairloop for finest level part
        P = data.ParticleDat(ncomp=3, dtype=dtype)
        Q = data.ParticleDat(ncomp=1, dtype=dtype)
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

        pair_kernel_src = """
        const double ipx = P.i[0] + {hex}; 
        const double ipy = P.i[1] + {hey}; 
        const double ipz = P.i[2] + {hez};
        const int icx = 100. + ipx*{lx};
        const int icy = 100. + ipy*{ly};
        const int icz = 100. + ipz*{lz};

        const double jpx = P.j[0] + {hex}; 
        const double jpy = P.j[1] + {hey}; 
        const double jpz = P.j[2] + {hez};
        const int jcx = 100. + jpx*{lx};
        const int jcy = 100. + jpy*{ly};
        const int jcz = 100. + jpz*{lz};
        
        const int dx = icx - jcx;
        const int dy = icy - jcy;
        const int dz = icz - jcz;

        int dr2 = dx*dx + dy*dy + dz*dz;
        
        {FREE_SPACE}

        const double mask = ((dr2 > 3) ? 0.0 : 0.5) * \
            maskx*masky*maskz;
        
        const double rx = P.j[0] - P.i[0];
        const double ry = P.j[1] - P.i[1];
        const double rz = P.j[2] - P.i[2];

        const double r2 = rx*rx + ry*ry + rz*rz;
        const double r = sqrt(r2);
        if (mask > 0) {{
        printf("---------------------------\\n");
        printf("KERNEL: %f %f %d \\n", mask, r, dr2);
        printf("\t%d\t%d\t%d \\n", dx, dy, dz);
        printf("\tI\t%f\t%f\t%f\t%d\t%d\t%d\\n", P.i[0], P.i[1], P.i[2], icx, icy, icz);
        printf("\tJ\t%f\t%f\t%f\t%d\t%d\t%d\\n", P.j[0], P.j[1], P.j[2], jcx, jcy, jcz);
        }}
        
        PHI[0] += mask * Q.i[0] * Q.j[0] / r;
        """.format(**{
            'hex': self.domain.extent[0] * 0.5,
            'hey': self.domain.extent[1] * 0.5,
            'hez': self.domain.extent[2] * 0.5,
            'lx': ns / self.domain.extent[0],
            'ly': ns / self.domain.extent[1],
            'lz': ns / self.domain.extent[2],
            'FREE_SPACE': free_space_mod
        })
        pair_kernel = kernel.Kernel('fmm_pairwise', code=pair_kernel_src, 
            headers=(kernel.Header('math.h'),))

        cell_by_cell = True
        if cell_by_cell:
            PL = pairloop.CellByCellOMP
            max_radius = 2. * (maxe + shell_width)
        else:
            PL = pairloop.PairLoopNeighbourListNSOMP
            max_radius = 1. * ((((maxe+shell_width)*2.)**2.)*3.)**0.5

        self._pair_loop = PL(
            kernel=pair_kernel,
            dat_dict={
                'P':P(access.READ),
                'Q':Q(access.READ),
                'PHI':self.particle_phi(access.INC_ZERO)
            },
            shell_cutoff=max_radius
        )
 
        self._int_list = list(range(self.R))
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

        self.timer_halo = opt.Timer(runtime.TIMER)
        self.timer_down = opt.Timer(runtime.TIMER)
        self.timer_up = opt.Timer(runtime.TIMER)

        self.execution_count = 0

        # threading
        self._async_thread = None

    def _update_opt(self):
        p = opt.PROFILE
        b = self.__class__.__name__ + ':'
        p[b+'contrib'] = self.timer_contrib.time()
        p[b+'extract'] = self.timer_extract.time()
        p[b+'mtm'] = self.timer_mtm.time()
        p[b+'mtl'] = self.timer_mtl.time()
        p[b+'ltl'] = self.timer_ltl.time()
        p[b+'local'] = self.timer_local.time()
        p[b+'halo'] = self.timer_halo.time()
        p[b+'down'] = self.timer_down.time()
        p[b+'up'] = self.timer_up.time()
        p[b+'exec_count'] = self.execution_count

    def _compute_local_interaction(self, positions, charges):
        self.timer_local.start()
        self._pair_loop.execute(
            dat_dict = {
                'P':positions(access.READ),
                'Q':charges(access.READ),
                'PHI':self.particle_phi(access.INC_ZERO)
            }
        )
        self.timer_local.pause()
        return self.particle_phi[0]

    def re_lm(self, l,m): return (l**2) + l + m


    def im_lm(self, l,m): return (l**2) + l +  m + self.L**2

    def __call__(self, positions, charges, forces=None, async=False):

        self._compute_cube_contrib(positions, charges)
        for level in range(self.R - 1, 0, -1):

            self._level_call_async(self._translate_m_to_m, level, async)
            self._halo_exchange(level)
            self._level_call_async(self._translate_m_to_l, level, async)
            self._fine_to_coarse(level)
            self.tree_parent[level][:] = 0.0

        self._join_async()


        if self.free_space:
            self.tree_parent[0][:] = 0.0
            self.tree_plain[0][:] = 0.0
            self.tree_parent[1][:] = 0.0
            self.tree_plain[1][:] = 0.0
        else:
            self.tree_parent[0][:] = 0.0
            self.tree_plain[0][:] = 0.0

            #self._compute_periodic_boundary()
            #print(self.tree_parent[1][:])


        for level in range(1, self.R):

            self._translate_l_to_l(level)
            self._coarse_to_fine(level)

        #for level in range(self.R):
        #    print(level, 60*'-')
        #    print(self.tree_halo[level][:,:,:,0])

        phi_extract = self._compute_cube_extraction(positions, charges)
        phi_near = self._compute_local_interaction(positions, charges)

        self._update_opt()

        print("Near:", phi_near, "Far:", phi_extract)

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


    def _compute_periodic_boundary(self):
        lsize = self.tree[1].parent_local_size
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

    def _compute_cube_contrib(self, positions, charges):

        self.timer_contrib.start()
        ns = self.tree.entry_map.cube_side_count
        cube_side_counts = np.array((ns, ns, ns), dtype=UINT64)
        if self._thread_allocation.size < self._tcount * \
                positions.npart_local + 1:
            self._thread_allocation = np.zeros(
                int(self._tcount*(positions.npart_local*1.1 + 1)),dtype=INT32)
        else:
            self._thread_allocation[:self._tcount:] = 0


        err = self._contribution_lib(
            INT64(self.L),
            UINT64(positions.npart_local),
            INT32(self._tcount),
            _check_dtype(positions, REAL),
            _check_dtype(charges, REAL),
            _check_dtype(self.domain.extent, REAL),
            _check_dtype(self.entry_data.local_offset, UINT64),
            _check_dtype(self.entry_data.local_size, UINT64),
            _check_dtype(cube_side_counts, UINT64),
            _check_dtype(self.entry_data.data, REAL),
            _check_dtype(self._thread_allocation, INT32)
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))

        #self.tree_halo[self.R-1][2:-2:, 2:-2:, 2:-2:, :] = \
        #    self.entry_data[:,:,:,:]
        self.tree_halo[self.R-1][2:-2:, 2:-2:, 2:-2:, :] = 0.0
        self.entry_data.add_onto(self.tree_halo)

        self.timer_contrib.pause()

    def _compute_cube_extraction(self, positions, charges):

        self.timer_extract.start()
        ns = self.tree.entry_map.cube_side_count
        cube_side_counts = np.array((ns, ns, ns), dtype=UINT64)
        '''
        const INT64 nlevel,
        const UINT64 npart,
        const INT32 thread_max,
        const REAL * RESTRICT position, 
        const REAL * RESTRICT charge,
        const REAL * RESTRICT boundary, 
        const UINT64 * RESTRICT cube_offset,  // zyx (slowest to fastest)
        const UINT64 * RESTRICT cube_dim,     // as above
        const UINT64 * RESTRICT cube_side_counts,   // as above
        const REAL * RESTRICT local_moments,
        REAL * RESTRICT phi_data,              // lexicographic
        const INT32 * RESTRICT thread_assign
        '''

        phi = REAL(0)
        #self.entry_data[:,:,:,:] = self.tree_plain[self.R-1][:,:,:,:]
        self.entry_data.extract_from(self.tree_plain)

        err = self._extraction_lib(
            INT64(self.L),
            UINT64(positions.npart_local),
            INT32(self._tcount),
            _check_dtype(positions, REAL),
            _check_dtype(charges, REAL),
            _check_dtype(self.domain.extent, REAL),
            _check_dtype(self.entry_data.local_offset, UINT64),
            _check_dtype(self.entry_data.local_size, UINT64),
            _check_dtype(cube_side_counts, UINT64),
            _check_dtype(self.entry_data.data, REAL),
            ctypes.byref(phi),
            _check_dtype(self._thread_allocation, INT32)
        )
        if err < 0: raise RuntimeError('Negative return code: {}'.format(err))

        red_re = mpi.all_reduce(np.array((phi.value)))

        self.timer_extract.pause()
        return red_re


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
        '''
        const UINT32 * RESTRICT dim_parent,     // slowest to fastest
        const UINT32 * RESTRICT dim_child,      // slowest to fastest
        REAL * RESTRICT moments_child,
        const REAL * RESTRICT moments_parent,
        const REAL * RESTRICT ylm,
        const REAL * RESTRICT alm,
        const REAL * RESTRICT almr,
        const REAL * RESTRICT i_array,
        const REAL radius,
        const INT64 nlevel
        '''
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

        if self.free_space:
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

        #print(self.tree_halo[level][:,:,:,0])
        self.timer_halo.pause()

    def _translate_m_to_l(self, level):
        """

        """
        '''
        int translate_mtl(
	    const UINT32 * RESTRICT dim_child,      // slowest to fastest
	    const REAL * RESTRICT multipole_moments,
	    REAL * RESTRICT local_moments,
	    const REAL * RESTRICT phi_data,
	    const REAL * RESTRICT theta_data,
	    const REAL * RESTRICT alm,
	    const REAL * RESTRICT almr,
	    const REAL * RESTRICT i_array,
	    const REAL radius,
	    const INT64 nlevel,
	    const INT32 * RESTRICT int_list,
	    const INT32 * RESTRICT int_tlookup,
	    const INT32 * RESTRICT int_plookup,
	    const double * RESTRICT int_radius
        )
        '''

        if self.tree[level].local_grid_cube_size is None:
            return

        self.timer_mtl.start()
        self.tree_plain[level][:] = 0.0

        radius = self.domain.extent[0] / \
                 self.tree[level].ncubes_side_global

        #print("MTL in", self.tree_halo[level][:,:,:,0])

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
            _check_dtype(self._int_radius, ctypes.c_double)
        )
        #print("MTL out", self.tree_plain[level][:,:,:,0])

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


    def _compute_g(self):

        #print("G START ============================================")

        ncomp = ((self.L * 2)**2) * 2

        terms = np.zeros(ncomp, dtype=self.dtype)

        min_len = min(
            self.domain.extent[0],
            self.domain.extent[1],
            self.domain.extent[2]
        )

        for lx in range(2, self.L*2, 2):
            rc, vc, kappa = self._compute_parameters(lx)

            rc*=3

            kappa2 = kappa*kappa
            maxt = int(math.ceil(rc/min_len))
            iterset = range(-1 * maxt, maxt+1)
            if len(iterset) < 4: print("Warning, small real space cutoff.")

            #print(lx, rc, vc, kappa, maxt)

            for tx in itertools.product(iterset, iterset, iterset):
                dispt = self._image_to_sph(tx)

                nd1 = abs(tx[0]) > 1 or abs(tx[1]) > 1 or abs(tx[2]) > 1


                if dispt[0] <= rc and nd1:
                    #print(tx, dispt)
                    #print("\t\ttx", tx,
                    #    "\tn+1/2", lx + 0.5,
                    #    "\tk2", kappa2,
                    #    "\tk2r2", kappa2*dispt[0] * dispt[0],
                    #    "\tres", gammaincc(lx + 0.5, kappa2*dispt[0] * dispt[0])
                    #)
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
                        #im_exp = np.sin(mx * dispt[1]) * val


                        assert abs(scipy_p[mxi].imag) < 10.**-16

                        coeff = scipy_p[mxi].real * radius_coeff * \
                                gammaincc(lx + 0.5, kappa2radius2)
                        #print(lx, mx, scipy_p[mxi].real * radius_coeff * \
                        #        gammaincc(lx + 0.5, kappa2radius2))

                        terms[self.re_lm(lx, mx)] += coeff * re_exp
                        # imag part is zero (indexing is wrong here)
                        #terms[self.im_lm(lx, mx)] += coeff * im_exp

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
        ivolume = 1./np.dot(lx, np.cross(ly, lz))

        gx = np.cross(ly,lz)*ivolume #* 2. * math.pi
        gy = np.cross(lz,lx)*ivolume #* 2. * math.pi
        gz = np.cross(lx,ly)*ivolume #* 2. * math.pi

        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)

        #print(gx)
        #print(self.domain.extent[0])
        #print(gyl)
        #print(gzl)

        for lx in range(2, self.L*2, 2):

            rc, vc, kappa = self._compute_parameters(lx)

            #print(lx, rc, vc, kappa)

            kappa2 = kappa * kappa
            mpi2okappa2 = -1.0 * (math.pi ** 2.) / kappa2

            nmax_x = int(ceil(vc/gxl))
            nmax_y = int(ceil(vc/gyl))
            nmax_z = int(ceil(vc/gzl))

            igamma = rgamma(lx + 0.5)

            #print("lx", lx, "\trgamma:", igamma)

            #print("nmax_x", nmax_x, gx, vc)

            for hxi in itertools.product(range(-1*nmax_z, nmax_z+1),
                                          range(-1*nmax_y, nmax_y+1),
                                          range(-1*nmax_x, nmax_x+1)):

                hx = hxi[0]*gz + hxi[1]*gy + hxi[2]*gx
                dispt = self._cart_to_sph(hx)


                if 10.**-10 < dispt[0] <= vc:

                    exp_coeff = math.exp(mpi2okappa2 * dispt[0] * dispt[0])*\
                                ivolume * igamma



                    mval = list(range(-1*lx, 1, 2)) + list(range(2, lx+1, 2))
                    mxval = [abs(mx) for mx in mval]

                    scipy_p = lpmv(mxval, lx, math.cos(dispt[2]))

                    vhnm2 = ((dispt[0] ** (lx - 2.)) * ((0 + 1.j) ** lx) * \
                            (math.pi ** (lx - 0.5))).real

                    #print("lx", lx,
                    #      "hx:", hx,
                    #      "exp_coeff", math.exp(mpi2okappa2 * dispt[0] * dispt[0]),
                    #      "v**2", dispt[0]**2.,
                    #      "vc2", vc*vc,
                    #      "vhnm2", vhnm2
                    #      )

                    for mxi, mx in enumerate(mval):
                        val = math.sqrt(float(math.factorial(
                            lx - abs(mx)))/math.factorial(lx + abs(mx)))
                        re_exp = np.cos(mx * dispt[1]) * val

                        assert abs(scipy_p[mxi].imag) < 10.**-16
                        sph_nm = re_exp * scipy_p[mxi].real

                        contrib = vhnm2 * sph_nm * exp_coeff

                        #print(lx, mx, "\trgamma, vhnm2:", vhnm2, "\t\t\tsph_nm:", sph_nm,
                        #      "\texp_coeff", exp_coeff, "\tcontrib", contrib)

                        #print("lx", lx, "contrib", contrib, "sph_nm", sph_nm,
                        #     "vhnm2", vhnm2, "exp_coeff", exp_coeff)
                        terms[self.re_lm(lx, mx)] += contrib.real


                        # imag part is zero (also indexing is wrong here)
                        #terms[self.im_lm(lx, mx)] += contrib.imag

            #for mx in list(range(-1*lx, 1, 2)) + list(range(2, lx+1, 2)):
            #    print("FINAL, lx", lx, "mx", mx, "val", terms[self.re_lm(lx, mx)] )


        #print("F END ============================================")
        return terms










