from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

from ppmd.coulomb.wigner import *
from ppmd.coulomb.wigner import _wigner_engine
from ppmd.coulomb import wigner

from transforms3d.euler import mat2euler
from scipy.special import sph_harm, lpmv
import time

from math import *

from common import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = False
SHARED_MEMORY = 'omp'

@pytest.fixture(
    scope="module",
    params=(-6.28318531, -4.71238898, -3.14159265, -1.57079633,  0.,
            1.57079633,  3.14159265,  4.71238898,  6.28318531 )
)
def phi_set(request):
    return request.param
@pytest.fixture(
    scope="module",
    params=(-6.28318531, -4.71238898, -3.14159265, -1.57079633,  0.,
            1.57079633,  3.14159265,  4.71238898,  6.28318531 )
)
def theta_set(request):
    return request.param

angle_set = (-6.28318531, -4.71238898, -3.14159265, -1.57079633,  0.,
            1.57079633,  3.14159265,  4.71238898,  6.28318531 )



# test forward rotate
def test_matmul_1():
    R = 3
    L = 10
    free_space = True
    N = 2
    E = 4.

    A = state.State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=1234)
    A.P[:] = rng.uniform(low=-0.4999*E, high=0.49999*E, size=(N,3))
    A.Q[:] = rng.uniform(low=-1., high=1., size=(N,1))
    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias

    Q = np.sum(A.Q[:]**2.)

    A.scatter_data_from(0)

    fmm = PyFMM(domain=A.domain, r=R, l=L, free_space=free_space
                , _debug=True)

    for alpha, beta in itertools.product(angle_set, angle_set):
        #alpha, beta = 0.0, 1.0


        exp_re = np.zeros(L-1, dtype=REAL)
        exp_im = np.zeros(L-1, dtype=REAL)
        for mxi, mx in enumerate(range(1, L)):
            me = cmath.exp(1.j * mx * alpha)
            exp_re[mxi] = me.real
            exp_im[mxi] = me.imag

        wp, wm = _wigner_engine(L, beta, eps_scaled=True)
        wpb, wmb = _wigner_engine(L, -beta, eps_scaled=True)

        
        # rotate moments forward using matvec
        lib_all_forw = fmm._translate_mtlz2_lib[
            'wrapper_rotate_moments_forward']
        lib_all_back = fmm._translate_mtlz2_lib[
            'wrapper_rotate_moments_backward']

        ncall = L*L
        re_xall = np.zeros(ncall, dtype=REAL)
        im_xall = np.zeros(ncall, dtype=REAL)

        re_tmp0 = np.zeros(ncall, dtype=REAL)
        im_tmp0 = np.zeros(ncall, dtype=REAL)

        re_oall = np.zeros(ncall, dtype=REAL)
        im_oall = np.zeros(ncall, dtype=REAL)

        re_tmp1 = np.zeros(ncall, dtype=REAL)
        im_tmp1 = np.zeros(ncall, dtype=REAL)
        
        
        BS = fmm.mtl_block_size
        blib_forw = fmm._translate_mtlz2_lib[
        'wrapper_blocked_forw_matvec'] 
        
        blib_back = fmm._translate_mtlz2_lib[
        'wrapper_blocked_back_matvec'] 
        # vars to pass into forward rotate
        bncall = BS * ncall
        bstride = 2*ncall
        bre_xall = np.zeros((BS, bstride), dtype=REAL)
        
        bre_xall[:] = rng.uniform(low=-1, high=1., size=(BS,bstride))

        bre_tmp0 = np.zeros(2*bncall, dtype=REAL)
        bre_oall = np.zeros(2*bncall, dtype=REAL)
        bre_yall = np.zeros(2*bncall, dtype=REAL)
        
        x_ptrs = np.zeros(BS, ctypes.c_void_p)
        for bx in range(BS):
            x_ptrs[bx] = bre_xall[bx].ctypes.data

        # test for p = 3
        blib_forw(
            INT64(BS),
            INT64(L),
            exp_re.ctypes.get_as_parameter(),
            exp_im.ctypes.get_as_parameter(),
            wp.ctypes.get_as_parameter(),
            wpb.ctypes.get_as_parameter(),
            x_ptrs.ctypes.get_as_parameter(),
            bre_tmp0.ctypes.get_as_parameter(),
            bre_oall.ctypes.get_as_parameter()
        )
        bre_tmp0[:] = 0
        bre_yall[:] = 0
        blib_back(
            INT64(BS),
            INT64(L),
            exp_re.ctypes.get_as_parameter(),
            exp_im.ctypes.get_as_parameter(),
            wp.ctypes.get_as_parameter(),
            wpb.ctypes.get_as_parameter(),
            bre_oall.ctypes.get_as_parameter(),
            bre_tmp0.ctypes.get_as_parameter(),
            bre_yall.ctypes.get_as_parameter()
        )

        for blkx in range(BS):
            s = blkx*ncall;
            e = s + ncall
            tmp = bre_yall

            back_err = np.linalg.norm(bre_xall[blkx, :ncall:] - tmp[s:e:], np.inf)
            assert back_err < 1*10.**-13
            back_err = np.linalg.norm(bre_xall[blkx, ncall::] - \
                    tmp[s+bncall:e+bncall:], np.inf)
            assert back_err < 1*10.**-13


        for blkx in range(BS):

            re_xall[:] = bre_xall[blkx, :ncall:]
            im_xall[:] = bre_xall[blkx, ncall::]

            lib_all_forw(
                INT32(L),
                exp_re.ctypes.get_as_parameter(),
                exp_im.ctypes.get_as_parameter(),
                wp.ctypes.get_as_parameter(),
                re_xall.ctypes.get_as_parameter(),
                im_xall.ctypes.get_as_parameter(),
                re_tmp0.ctypes.get_as_parameter(),
                im_tmp0.ctypes.get_as_parameter(),
                re_oall.ctypes.get_as_parameter(),
                im_oall.ctypes.get_as_parameter()
            )
            
            s = blkx*ncall;
            e = s + ncall

            err_re = np.linalg.norm(re_oall[:] - bre_oall[s:e:], np.inf)
            assert err_re < 10.**-14
            err_im = np.linalg.norm(im_oall[:] - bre_oall[bncall+s:bncall+e:], np.inf)
            assert err_im < 10.**-14

    
def test_blocked_mtl_z_1():
    R = 3
    L = 10
    free_space = True
    N = 2
    E = 4.

    A = state.State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    A.P = data.PositionDat(ncomp=3)

    rng = np.random.RandomState(seed=1234)


    A.scatter_data_from(0)

    fmm = PyFMM(domain=A.domain, r=R, l=L, free_space=free_space
                , _debug=True)
    

    BS = fmm.mtl_block_size
    ncall = L*L
    bncall = BS * ncall
    bstride = 2*ncall

    bre_oall = np.zeros(2*bncall, dtype=REAL)
    bre_oall[:] = rng.uniform(low=-1, high=1., size=2*bncall)

    bre_lall = np.zeros(2*bncall, dtype=REAL)

    blib_mtlz = fmm._translate_mtlz2_lib['wrapper_blocked_mtl_z'] 

    radius = 1.0
    blib_mtlz(
        INT64(BS),
        INT64(L),
        REAL(radius),
        fmm._a.ctypes.get_as_parameter(),
        fmm._arn0.ctypes.get_as_parameter(),
        fmm._ipower_mtl.ctypes.get_as_parameter(),
        bre_oall.ctypes.get_as_parameter(),
        bre_lall.ctypes.get_as_parameter()
    )
    
    true_o = np.zeros(bstride)
    
    for bx in range(BS):
        s = bx*ncall
        e = s + ncall
        true_o[:ncall:] = bre_oall[s:e:]
        true_o[ncall::] = bre_oall[bncall + s : bncall + e : ]
        true_l = shift_z(L, radius, 0.0, true_o) 
        err_re = np.linalg.norm(true_l[:ncall:] - bre_lall[s:e:], np.inf)
        err_im = np.linalg.norm(true_l[ncall::] - bre_lall[bncall+s:bncall+e:], np.inf)
        assert err_re < 10.**-10
        assert err_im < 10.**-10






















