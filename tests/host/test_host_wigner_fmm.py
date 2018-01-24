from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np
from ctypes import *

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'), filename)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

from ppmd.coulomb.wigner import *

from transforms3d.euler import mat2euler
from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

def red(*input):
    try:
        from termcolor import colored
        return colored(*input, color='red')
    except Exception as e: return input
def green(*input):
    try:
        from termcolor import colored
        return colored(*input, color='green')
    except Exception as e: return input
def yellow(*input):
    try:
        from termcolor import colored
        return colored(*input, color='yellow')
    except Exception as e: return input

def red_tol(val, tol):
    if abs(val) > tol:
        return red(str(val))
    else:
        return green(str(val))


def tuple_it(*args, **kwargs):
    if len(kwargs) == 0:
        tx = args[0]
        return itertools.product(range(tx[0]), range(tx[1]), range(tx[2]))
    else:
        l = kwargs['low']
        h = kwargs['high']
        return itertools.product(range(l[0], h[0]),
                                 range(l[1], h[1]),
                                 range(l[2], h[2]))


def spherical(xyz):
    if type(xyz) is tuple or len(xyz.shape) == 1:
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

    #print("spherical", xyz, sph)
    return sph


def Hfoo(nx, mx):
    return math.sqrt(
        float(math.factorial(nx - abs(mx)))/math.factorial(nx + abs(mx))
    )

def Pfoo(nx, mx, x):
    if abs(mx) > abs(nx):
        return 0.0
    elif nx < 0:
        return Pfoo(-1*nx -1, mx, x)
    else:
        return lpmv(mx, nx, x)

def Yfoo(nx, mx, theta, phi):
    coeff = Hfoo(nx, mx)
    legp = lpmv(abs(mx), nx, math.cos(theta))

    assert abs(legp.imag) < 10.**-16

    return coeff * legp * cmath.exp(1.j * mx * phi)

def Hfoo(nx, mx):
    return math.sqrt(
        float(math.factorial(nx - abs(mx)))/math.factorial(nx + abs(mx))
    )

def Pfoo(nx, mx, x):
    if abs(mx) > abs(nx):
        return 0.0
    elif nx < 0:
        return Pfoo(-1*nx -1, mx, x)
    else:
        return lpmv(mx, nx, x)

def Yfoo(nx, mx, theta, phi):
    coeff = Hfoo(nx, mx)
    legp = lpmv(abs(mx), nx, math.cos(theta))

    assert abs(legp.imag) < 10.**-16

    return coeff * legp * cmath.exp(1.j * mx * phi)

def Afoo(n, m):
    if n - m < 0:
        return 0.0
    if n + m < 0:
        return 0.0

    return ((-1.)**n)/float(
    math.sqrt(math.factorial(n - m) * math.factorial(n + m)))

def Ifoo(k, m): return ((1.j) ** (abs(k-m) - abs(k) - abs(m)))


def shift_normal(L, radius, theta, phi, moments):

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + L**2

    out = np.zeros_like(moments)
    # translate
    for jx in range(L):
        for kx in range(-1*jx, jx+1):

            for nx in range(L):
                for mx in range(-1*nx, nx+1):

                    Onm = moments[re_lm(nx, mx)] + (1.j) * \
                        moments[im_lm(nx, mx)]

                    Onm *= (1.j)**(abs(kx-mx) - abs(kx) - abs(mx))
                    Onm *= Afoo(nx, mx)
                    Onm *= Afoo(jx, kx)
                    Onm *= Yfoo(jx+nx, mx-kx, theta, phi)
                    Onm *= (-1.)**nx
                    Onm /= Afoo(jx+nx, mx-kx)
                    Onm /= radius**(jx+nx+1.)

                    out[re_lm(jx, kx)] += Onm.real
                    out[im_lm(jx, kx)] += Onm.imag

    return out



def shift_z(L, radius, theta, moments):

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + L**2

    out = np.zeros_like(moments)
    # translate
    for jx in range(L):
        for kx in range(-1*jx, jx+1):

            for nx in range(L):
                Onm = 0.0 if abs(kx) > nx else 1.0

                Onm *= moments[re_lm(nx, kx)] + (1.j) * \
                    moments[im_lm(nx, kx)]

                Onm *= (-1.)**kx
                Onm *= Afoo(nx, kx)
                Onm *= Afoo(jx, kx)
                Onm *= Yfoo(jx+nx, 0, theta, 0)
                #Onm *= ct**(nx+jx)
                Onm *= (-1.)**nx
                Onm /= Afoo(jx+nx, 0)
                Onm /= radius**(jx+nx+1.)

                out[re_lm(jx, kx)] += Onm.real
                out[im_lm(jx, kx)] += Onm.imag

    return out

def rotate_z(phi):
    return np.array((
        (math.cos(phi), -1.* math.sin(phi)  , 0.0),
        (math.sin(phi),      math.cos(phi)  , 0.0),
        (          0.0,               0.0   , 1.0)
    ))

def rotate_y(theta):
    return np.array((
        (math.cos(theta)  ,   0.0, math.sin(theta)),
        (            0.0  ,   1.0,           0.0),
        (-1.*math.sin(theta),   0.0, math.cos(theta))
    ))

def matvec(A,b):
    N = max(b.shape[:])
    out = np.zeros_like(b)
    for rx in range(N):
        out[rx] = np.dot(A[rx, :], b[:])
    return out

def eps_m(m):
    if m < 0: return 1.0
    return (-1.)**m

def rotate_moments(L, alpha, beta, gamma, moments):
    return rotate_moments_matrix(L,alpha,beta,gamma,moments)

def rotate_moments_matrix(L, alpha, beta, gamma, moments):
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + L**2

    out = np.zeros_like(moments)
    for nx in range(L):

        rotmatrix = R_zyz(nx, alpha=alpha, beta=beta, gamma=gamma)
        vec = np.zeros(2*nx + 1, dtype=np.complex)

        re_start = re_lm(nx, -nx)
        re_end = re_lm(nx,nx)+1

        im_start = im_lm(nx, -nx)
        im_end = im_lm(nx,nx)+1

        vec.real[:] = moments[re_start:re_end:]
        vec.imag[:] = moments[im_start:im_end:]

        ab = np.matmul(rotmatrix, vec)

        out[re_start:re_end:] = ab.real[:]
        out[im_start:im_end:] = ab.imag[:]


    return out


def test_fmm_cplx_matvec_1():
    R = 2
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
            c_int32(nval),
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
        c_int32(nterms),
        pointers_real.ctypes.get_as_parameter(),
        pointers_imag.ctypes.get_as_parameter(),
        moments.ctypes.get_as_parameter(),
        moments[nterms*nterms:].view().ctypes.get_as_parameter(),
        bvec.ctypes.get_as_parameter(),
        bvec[nterms*nterms:].view().ctypes.get_as_parameter()
    )

    err = np.linalg.norm(correct[:] - bvec[:], np.inf)
    assert err < tol




def shift_z(L, radius, theta, moments):

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + L**2

    out = np.zeros_like(moments)

    # translate
    for jx in range(L):
        for kx in range(-1*jx, jx+1):

            for nx in range(abs(kx), L):
                Onm = 1.0

                Onm *= moments[re_lm(nx, kx)] + (1.j) * moments[im_lm(nx, kx)]
                Onm *= (-1.)**kx
                Onm *= Afoo(nx, kx)
                Onm *= Afoo(jx, kx)
                Onm *= Yfoo(jx+nx, 0, theta, 0)
                Onm *= (-1.)**nx
                Onm /= Afoo(jx+nx, 0)
                Onm /= radius**(jx+nx+1.)

                out[re_lm(jx, kx)] += Onm.real
                out[im_lm(jx, kx)] += Onm.imag

    return out


def test_fmm_translate_1():
    R = 2
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

    tmp_space = np.zeros(ncomp, dtype=fmm.dtype)

    fmm._translate_mtlz_lib['mtl_z_wrapper'](
        ctypes.c_int64(nterms),
        fmm.dtype(radius),
        moments.ctypes.get_as_parameter(),
        pointers_real.ctypes.get_as_parameter(),
        pointers_imag.ctypes.get_as_parameter(),
        back_pointers_real.ctypes.get_as_parameter(),
        back_pointers_imag.ctypes.get_as_parameter(),
        fmm._a.ctypes.get_as_parameter(),
        fmm._ar.ctypes.get_as_parameter(),
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


def test_fmm_translate_2():
    R = 2
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




def test_fmm_translate_3():
    R = 3
    eps = 10.**-8
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


    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space
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

    t0 = time.time()
    fmm._translate_m_to_l_cart(level)
    t1 = time.time()

    correct = np.copy(fmm.tree_plain[level][:,:,:,:])

    fmm.tree_plain[level][:,:,:,:] = 0

    t2 = time.time()
    fmm._translate_m_to_l(level)
    t3 = time.time()

    zmtl = np.copy(fmm.tree_plain[level][:,:,:,:])


    correct = correct.ravel()
    zmtl = zmtl.ravel()
    err = np.linalg.norm(correct - zmtl, np.inf)

    print("\n")
    print("TIME CART: \t", t1 - t0)
    print("TIME ZAXI: \t", t3 - t2)

    tol = 10.**-13
    assert err < tol














