from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np
from ctypes import *

#from ppmd_vis import plot_spheres

import itertools


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

from ppmd.coulomb.wigner import *
from ppmd.coulomb.wigner import _wigner_engine

from transforms3d.euler import mat2euler
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = False
SHARED_MEMORY = 'omp'

from common import *

INT64 = c_int64
REAL = c_double

@pytest.mark.skipif("MPISIZE>1")
def test_fmm_cplx_matvec_1():
    R = 3
    eps = 10.**-2
    free_space = True
    N = 2
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space
                , _debug=True)

    tol = 10.**-14
    nterms = 30
    ncomp = (nterms**2)*2
    im_offset = nterms**2
    rng = np.random.RandomState(seed=1234)

    moments = rng.uniform(low=-1.0, high=1.0, size=ncomp)
    bvec = np.zeros_like(moments)


    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    theta = 0.12*math.pi
    phi = 1.7

    alpha, beta, gamma = 0.0, theta, phi

    correct = rotate_moments(nterms, alpha, beta, gamma, moments)

    # test each p
    for lx in range(nterms):
        rmat = R_zyz(lx, alpha=alpha, beta=beta, gamma=gamma)

        nval = 2*lx +1
        re_m = np.zeros((nval, nval), dtype=fmm.dtype)
        im_m = np.zeros((nval, nval), dtype=fmm.dtype)

        re_m[:] = rmat.real
        im_m[:] = rmat.imag

        re_x = moments[re_lm(lx,-lx):re_lm(lx,lx)+1:].view()
        im_x = moments[im_lm(lx,-lx):im_lm(lx,lx)+1:].view()

        re_b = bvec[re_lm(lx,-lx):re_lm(lx,lx)+1:].view()
        im_b = bvec[im_lm(lx,-lx):im_lm(lx,lx)+1:].view()

        # use lib cplx matvec
        fmm._translate_mtlz_lib['rotate_p_moments_wrapper'](
            INT64(nval),
            re_m.ctypes.get_as_parameter(),
            im_m.ctypes.get_as_parameter(),
            re_x.ctypes.get_as_parameter(),
            im_x.ctypes.get_as_parameter(),
            re_b.ctypes.get_as_parameter(),
            im_b.ctypes.get_as_parameter()
        )

        err_re = np.linalg.norm(correct[re_lm(lx,-lx):re_lm(lx,lx)+1:] - re_b,
                                np.inf)
        err_im = np.linalg.norm(correct[im_lm(lx,-lx):im_lm(lx,lx)+1:] - im_b,
                                np.inf)
        assert err_re < tol
        assert err_im < tol

    # test for indexing error
    err = np.linalg.norm(correct[:] - bvec[:], np.inf)
    assert err < tol

    # storage to prevent matrices/pointer arrays going out of scope
    wigner_matrices = []
    # rotate all terms at once
    pointers_real, pointers_imag, matrices = Rzyz_set(
        p=nterms,
        alpha=alpha, beta=beta, gamma=gamma,
        dtype=fmm.dtype)
    # store the temporaries
    wigner_matrices.append(matrices)

    # use lib cplx matvec
    fmm._translate_mtlz_lib['rotate_moments_wrapper'](
        INT64(nterms),
        pointers_real.ctypes.get_as_parameter(),
        pointers_imag.ctypes.get_as_parameter(),
        moments.ctypes.get_as_parameter(),
        moments[nterms*nterms:].view().ctypes.get_as_parameter(),
        bvec.ctypes.get_as_parameter(),
        bvec[nterms*nterms:].view().ctypes.get_as_parameter()
    )

    err = np.linalg.norm(correct[:] - bvec[:], np.inf)
    assert err < tol

    fmm.free()



@pytest.mark.skipif("MPISIZE>1")
def test_fmm_translate_1():
    R = 3
    eps = 10.**-2
    free_space = True
    N = 2
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space
                , _debug=True)

    tol = 10.**-14
    nterms = fmm.L
    ncomp = (nterms**2)*2
    im_offset = nterms**2
    rng = np.random.RandomState(seed=1234)

    moments = rng.uniform(low=-1.0, high=1.0, size=ncomp)
    bvec = np.zeros_like(moments)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    radius = 5.
    theta = 0.0
    phi = 0.0

    alpha, beta, gamma = 0.0, theta, phi

     # storage to prevent matrices/pointer arrays going out of scope
    pointers_real, pointers_imag, matrices = Rzyz_set(
        p=nterms,
        alpha=alpha, beta=beta, gamma=gamma,
        dtype=fmm.dtype)

    back_pointers_real, back_pointers_imag, back_matrices = Rzyz_set(
        p=nterms,
        alpha=-gamma, beta=-beta, gamma=-alpha,
        dtype=fmm.dtype)

    tmp_space = np.zeros(2*ncomp, dtype=fmm.dtype)


    fmm._translate_mtlz_lib['mtl_z_wrapper'](
        ctypes.c_int64(nterms),
        fmm.dtype(radius),
        moments.ctypes.get_as_parameter(),
        pointers_real.ctypes.get_as_parameter(),
        pointers_imag.ctypes.get_as_parameter(),
        back_pointers_real.ctypes.get_as_parameter(),
        back_pointers_imag.ctypes.get_as_parameter(),
        fmm._a.ctypes.get_as_parameter(),
        fmm._arn0.ctypes.get_as_parameter(),
        fmm._ipower_mtl.ctypes.get_as_parameter(),
        bvec.ctypes.get_as_parameter(),
        tmp_space.ctypes.get_as_parameter()
    )

    correct = shift_z(nterms, radius, 0.0, moments)

    err = np.linalg.norm(bvec - correct, np.inf)

    if DEBUG:
        for nx in range(nterms):
            print("nx =", nx)
            for mx in range(-1*nx, nx+1):
                print("\t{: 2d} | {: .8f} {: .8f} | {: .8f} {: .8f}  || {: .8f} {: .8f}".format(mx,
                    correct[re_lm(nx, mx)], bvec[re_lm(nx, mx)],
                    correct[im_lm(nx, mx)], bvec[im_lm(nx, mx)],
                    moments[re_lm(nx, mx)], moments[im_lm(nx, mx)]))

        print("ERR:\t", red_tol(err, tol))

    assert err < tol

    fmm.free()


@pytest.mark.skipif("MPISIZE>1")
def test_fmm_translate_2():
    R = 3
    eps = 10.**-4
    free_space = True
    N = 2
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space
                , _debug=True)

    tol = 10.**-14
    nterms = fmm.L
    ncomp = (nterms**2)*2
    im_offset = nterms**2
    rng = np.random.RandomState(seed=1234)

    moments = rng.uniform(low=-10.0, high=10.0, size=ncomp)
    bvec = np.zeros_like(moments)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    radius = 2.
    theta = 1.1*math.pi
    phi = -1.1234

    alpha, beta, gamma = 0.0, theta, phi

     # storage to prevent matrices/pointer arrays going out of scope
    pointers_real, pointers_imag, matrices = Rzyz_set(
        p=nterms,
        alpha=alpha, beta=beta, gamma=gamma,
        dtype=fmm.dtype)

    back_pointers_real, back_pointers_imag, back_matrices = Rzyz_set(
        p=nterms,
        alpha=-gamma, beta=-beta, gamma=-alpha,
        dtype=fmm.dtype)

    tmp_space = np.zeros(ncomp*2, dtype=fmm.dtype)

    t0 = time.time()
    fmm._translate_mtlz_lib['mtl_z_wrapper'](
        ctypes.c_int64(nterms),
        fmm.dtype(radius),
        moments.ctypes.get_as_parameter(),
        pointers_real.ctypes.get_as_parameter(),
        pointers_imag.ctypes.get_as_parameter(),
        back_pointers_real.ctypes.get_as_parameter(),
        back_pointers_imag.ctypes.get_as_parameter(),
        fmm._a.ctypes.get_as_parameter(),
        fmm._arn0.ctypes.get_as_parameter(),
        fmm._ipower_mtl.ctypes.get_as_parameter(),
        bvec.ctypes.get_as_parameter(),
        tmp_space.ctypes.get_as_parameter()
    )
    t1 = time.time()


    forward_rot = rotate_moments(nterms, alpha=alpha, beta=beta, gamma=gamma,
                                 moments=moments)
    z_mtl = shift_z(nterms, radius, 0.0, forward_rot)

    correct = rotate_moments(nterms, alpha=-gamma, beta=-beta, gamma=-alpha,
                             moments=z_mtl)

    err = np.linalg.norm(bvec - correct, np.inf)

    if DEBUG:
        for nx in range(nterms):
            print("nx =", nx)
            for mx in range(-1*nx, nx+1):
                print("\t{: 2d} | {: .8f} {: .8f} | {: .8f} {: .8f}  || {: .8f} {: .8f}".format(mx,
                    correct[re_lm(nx, mx)], bvec[re_lm(nx, mx)],
                    correct[im_lm(nx, mx)], bvec[im_lm(nx, mx)],
                    moments[re_lm(nx, mx)], moments[im_lm(nx, mx)]))

    print("ERR:\t", red_tol(err, tol), "\tTIME:\t", t1-t0)

    assert err < tol
    fmm.free()



def test_fmm_translate_3():
    R = 3
    
    eps = 10.**-3
    L = 20

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


    fmm = PyFMM(domain=A.domain, r=R, eps=None, l=L, free_space=free_space
                , _debug=True)

    positions = A.P
    charges = A.Q

    fmm.entry_data.zero()
    fmm.tree_plain.zero()
    fmm.tree_halo.zero()
    fmm.tree_parent.zero()

    fmm._check_aux_dat(positions)

    fmm._compute_cube_contrib(positions, charges,
                               positions.group._fmm_cell)

    level = fmm.R-1

    sh = fmm.tree_halo[level][:,:,:,:].shape
    
    fmm.tree_halo[level][:,:,:,:] = rng.uniform(low=-2.0, high=2.0, size=sh)


    # break values
    #fmm.tree_halo[level][:,:,:,:] = 0.0
    #fmm.tree_halo[level][0,2,3,0] = 0.42
    #print('INPUT START')
    #print(fmm.tree_halo[level][0,2,3,:4])
    #print('INPUT END')

    t0 = time.time()
    fmm._translate_m_to_l_cart(level)
    t1 = time.time()

    correct = np.copy(fmm.tree_plain[level][:,:,:,:])

    fmm.tree_plain[level][:,:,:,:] = 0

    t2 = time.time()
    fmm._translate_m_to_l(level)
    t3 = time.time()

    zmtl = np.copy(fmm.tree_plain[level][:,:,:,:])


    print("\n")
    print("TIME CART: \t", t1 - t0)
    print("TIME ZAXI: \t", t3 - t2)
    
    #print("L:", fmm.L)

    #print(correct[0,0,1,:5:])
    #print(zmtl[0,0,1,:5:])

    correct = correct.ravel()
    zmtl = zmtl.ravel()
    err = np.linalg.norm(correct - zmtl, np.inf)
    tol = 10.**-13
    assert err < tol
    #print("ERROR:", err)

    fmm.free()



@pytest.mark.skipif("MPISIZE>1")
def test_fmm_translate_split_1():
    R = 3
    eps = 10.**-4
    free_space = True
    N = 2
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space
                , _debug=True)

    tol = 10.**-14
    nterms = fmm.L
    ncomp = (nterms**2)*2
    im_offset = nterms**2
    rng = np.random.RandomState(seed=1234)

    moments = rng.uniform(low=-10.0, high=10.0, size=ncomp)
    bvec = np.zeros_like(moments)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    radius = 2.
    theta = -4.7
    phi = -2.6

    alpha, beta, gamma = 0.0, theta, phi

     # storage to prevent matrices/pointer arrays going out of scope
    pointers_real, pointers_imag, matrices = Rzyz_set(
        p=nterms,
        alpha=alpha, beta=beta, gamma=gamma,
        dtype=fmm.dtype)

    back_pointers_real, back_pointers_imag, back_matrices = Rzyz_set(
        p=nterms,
        alpha=-gamma, beta=-beta, gamma=-alpha,
        dtype=fmm.dtype)

    tmp_space = np.zeros(ncomp*2, dtype=fmm.dtype)
    """
    t0 = time.time()
    fmm._translate_mtlz_lib['mtl_z_wrapper'](
        ctypes.c_int64(nterms),
        fmm.dtype(radius),
        moments.ctypes.get_as_parameter(),
        pointers_real.ctypes.get_as_parameter(),
        pointers_imag.ctypes.get_as_parameter(),
        back_pointers_real.ctypes.get_as_parameter(),
        back_pointers_imag.ctypes.get_as_parameter(),
        fmm._a.ctypes.get_as_parameter(),
        fmm._arn0.ctypes.get_as_parameter(),
        fmm._ipower_mtl.ctypes.get_as_parameter(),
        bvec.ctypes.get_as_parameter(),
        tmp_space.ctypes.get_as_parameter()
    )
    t1 = time.time()
    """

    wp, wm = _wigner_engine(fmm.L, beta, eps_scaled=True)
    wpb, wmb = _wigner_engine(fmm.L, -beta, eps_scaled=True)
    exp_re = np.zeros(fmm.L-1, dtype=REAL)
    exp_im = np.zeros(fmm.L-1, dtype=REAL)
    for mxi, mx in enumerate(range(1, fmm.L)):
        me = cmath.exp(1.j * mx * alpha)
        exp_re[mxi] = me.real
        exp_im[mxi] = me.imag


    t0 = time.time()
    fmm._translate_mtlz2_lib['mtl_z_wrapper'](
        ctypes.c_int64(nterms),
        fmm.dtype(radius),
        moments.ctypes.get_as_parameter(),
        wp.ctypes.get_as_parameter(),
        wpb.ctypes.get_as_parameter(),
        exp_re.ctypes.get_as_parameter(),
        exp_im.ctypes.get_as_parameter(),
        fmm._a.ctypes.get_as_parameter(),
        fmm._arn0.ctypes.get_as_parameter(),
        fmm._ipower_mtl.ctypes.get_as_parameter(),
        bvec.ctypes.get_as_parameter(),
        tmp_space.ctypes.get_as_parameter()
    )
    t1 = time.time()




    forward_rot = rotate_moments(nterms, alpha=alpha, beta=beta, gamma=gamma,
                                 moments=moments)
    z_mtl = shift_z(nterms, radius, 0.0, forward_rot)

    correct = rotate_moments(nterms, alpha=-gamma, beta=-beta, gamma=-alpha,
                             moments=z_mtl)

    err = np.linalg.norm(bvec - correct, np.inf)

    if DEBUG:
        for nx in range(nterms):
            print("nx =", nx)
            for mx in range(-1*nx, nx+1):
                print("\t{: 2d} | {: .8f} {: .8f} | {: .8f} {: .8f}  || {: .8f} {: .8f}".format(mx,
                    correct[re_lm(nx, mx)], bvec[re_lm(nx, mx)],
                    correct[im_lm(nx, mx)], bvec[im_lm(nx, mx)],
                    moments[re_lm(nx, mx)], moments[im_lm(nx, mx)]))

    print("ERR:\t", red_tol(err, tol), "\tTIME:\t", t1-t0)

    assert err < tol

    fmm.free()








