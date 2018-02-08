from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'), filename)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

from ppmd.coulomb.wigner import *
from ppmd.coulomb import wigner

from transforms3d.euler import mat2euler
from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = False
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



def test_wigner_1():
    """Tests the recursive computation matches the jacobi method and
    some known values"""


    N = 10
    random_angles = np.random.uniform(low=0.0, high=2.*math.pi, size=N)
    tol = 10.**-12

    for beta in random_angles:
        if DEBUG:
            print("wigner_d |  1  0  0", wigner_d(1,0,0, beta),
                  math.cos(beta))
            print("wigner_d |  1  1  1", wigner_d(1,1,1, beta),
                  (1. + math.cos(beta)) * 0.5)
            print("wigner_d |  1  1 -1", wigner_d(1,1,-1, beta),
                  (1. - math.cos(beta))*0.5)
            print("wigner_d |  1  1  0", wigner_d(1,1,0, beta),
                  (-1.*math.sin(beta))/math.sqrt(2.))

        assert abs(wigner_d(1,0,0, beta)-\
                   math.cos(beta)) < tol
        assert abs(wigner_d(1,1,1, beta)-\
                   (1. + math.cos(beta)) * 0.5) < tol
        assert abs(wigner_d(1,1,-1, beta)-\
                   (1. - math.cos(beta))*0.5) < tol
        assert abs(wigner_d(1,1,0, beta)-\
              (-1.*math.sin(beta))/math.sqrt(2.)) < tol

    cos = math.cos
    sin = math.sin
    sqrt = math.sqrt

    j2 = (
        (2,2,2,lambda x:0.25*(1+cos(x))**2.),
        (2,2,1,lambda x:-0.5*sin(x)*(1+cos(x))),
        (2,2,0,lambda x:sqrt(3./8.)*(sin(x)**2.)),
        (2,2,-1,lambda x:-0.5*sin(x)*(1-cos(x))),
        (2,2,-2,lambda x:0.25*(1-cos(x))**2.),
        (2,1,1, lambda x: 0.5*(2*(cos(x)**2.) + cos(x) - 1.)),
        (2,1,0,lambda x:-1.*sqrt(3./8)*sin(2*x)),
        (2,1,-1,lambda x:0.5*(-2.*(cos(x)**2.) + cos(x) + 1)),
        (2,0,0, lambda x: 0.5*(3.*(cos(x)**2.) - 1 ))
    )

    for beta in random_angles:
        for jx in j2:
            assert abs(wigner_d(jx[0], jx[1], jx[2], beta) - jx[3](beta)) < tol


    beta= 0.1*math.pi
    jx = (1,0,0)


    for jx in j2:
        w0 = wigner_d_rec(jx[0],jx[1],jx[2],beta)
        w1 = wigner_d(jx[0],jx[1],jx[2],beta)
        if DEBUG:
            print(jx, "\t-------->\t",w0,"\t",w1,"\t",red_tol(abs(w1-w0),
                                                              10.**-14))

    L = 15
    random_angles = np.random.uniform(low=0.0, high=2.*math.pi, size=N)
    for beta in random_angles:
        for nx in range(0, L):
            for mpx in range(-1*nx, nx+1):
                for mx in range(-1*nx, nx+1):
                    w0 = wigner_d_rec(nx,mpx,mx,beta)
                    w1 = wigner_d    (nx,mpx,mx,beta)
                    err = abs(w0 - w1)
                    if DEBUG:
                        print(w0, w1, red_tol(err, tol))
                    assert err < tol

    L = 20
    beta = random_angles[-1]

    tr = 0.0
    tj = 0.0

    for nx in range(0, L):
        for mpx in range(-1*nx, nx+1):
            for mx in range(-1*nx, nx+1):
                t0 = time.time()
                w0 = wigner_d_rec(nx,mpx,mx,beta)
                tr += time.time() - t0

                t0 = time.time()
                w1 = wigner_d    (nx,mpx,mx,beta)
                tj += time.time() - t0

                err = abs(w0 - w1)
                if DEBUG:
                    print(w0, w1, red_tol(err, tol))
                assert err < tol

    if DEBUG:
        print("recusive:\t", tr)
        print("jacobi:\t", tj)


def test_wigner_2():
    """
    rotates random vectors forwards and backwards and checks the results are
    the same
    """

    tol = 10.**-12

    nangles = 4
    nterms = 10

    random_angles = np.random.uniform(low=0.0, high=2.0*math.pi, size=nangles)

    for nx in range(nterms):
        for beta in random_angles:

            original_values = np.random.uniform(
                low=-10.0, high=10, size=2*nx+1)
            rotated_values = np.zeros_like(original_values)
            rotated_values2 = np.zeros_like(original_values)

            # forward rotation
            for mpx in range(-1*nx, nx+1):
                tmp = 0.0
                for mx in range(-1*nx, nx+1):
                    tmp += wigner_d_rec(nx, mpx, mx, beta) * \
                           original_values[nx + mx]
                rotated_values[nx + mpx] = tmp

            # backward rotation
            for mpx in range(-1*nx, nx+1):
                tmp = 0.0
                for mx in range(-1*nx, nx+1):
                    tmp += wigner_d_rec(nx, mpx, mx, -1.0 * beta) * \
                           rotated_values[nx + mx]
                rotated_values2[nx + mpx] = tmp

            err = np.linalg.norm(rotated_values2 - original_values, np.inf)
            assert err < tol




def test_wigner_3():
    """
    rotates random vectors forwards and backwards and checks the results are
    the same. Uses rotate_moments. rotates through two angles
    """

    tol = 10.**-12

    nangles = 1
    nterms = 20


    ncomp = (nterms**2)*2
    angles = np.random.uniform(low=0.0, high=2.0*math.pi, size=(nangles, 2))

    #for anglex in range(nangles):

    orig = np.random.uniform(low=-10, high=10, size=ncomp)


    forw_ro = rotate_moments(nterms,   0.0,  0.1*math.pi, 0.0, orig)
    back_ro = rotate_moments(nterms,   0.0, -0.1*math.pi, 0.0, forw_ro)
    err = np.linalg.norm(orig - back_ro, np.inf)
    assert err < tol

    forw_ro = rotate_moments(nterms,   0.1,  0., 0.0, orig)
    back_ro = rotate_moments(nterms,  -0.1,  0., 0.0, forw_ro)
    err = np.linalg.norm(orig - back_ro, np.inf)
    assert err < tol

    forw_ro = rotate_moments(nterms,   0.1,  0.1*math.pi, 0.0, orig)
    back_ro = rotate_moments(nterms,   0.0, -0.1*math.pi, -0.1, forw_ro)

    err = np.linalg.norm(orig - back_ro, np.inf)
    assert err < tol



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


def test_rotatation_matrices_1():

    tol = 10.**-15

    x = np.array((1.0, 0.0, 0.0))
    y = np.array((0.0, 1.0, 0.0))
    z = np.array((0.0, 0.0, 1.0))

    assert np.linalg.norm(matvec(rotate_z(math.pi*0.5), x) - y, np.inf) < tol
    assert np.linalg.norm(matvec(rotate_z(math.pi), x) + x, np.inf) < tol
    assert np.linalg.norm(matvec(rotate_z(0.1353), z) - z, np.inf) < tol

    assert np.linalg.norm(matvec(rotate_y(0.5*math.pi), z) - x, np.inf) < tol
    assert np.linalg.norm(matvec(rotate_y(1.5*math.pi), z) + x, np.inf) < tol

@pytest.mark.skipif("DEBUG==False")
def test_print_1():
    """
    rotates random vectors forwards and backwards and checks the results are
    the same. Uses rotate_moments. rotates through two angles
    """

    tol = 10.**-12
    N = 4

    nangles = 1
    nterms = 4

    ncomp = (nterms**2)*2
    rng = np.random.RandomState(seed=1234)

    P = rng.uniform(low=-1., high=1., size=(N,3))

    P[0,:] = (1.0, 0.0, 0.0)
    P[1,:] = (0.0, 0.0, 1.0)
    P[2,:] = (-1.0, 0.0, 0.0)
    P[3,:] = (0.0, 0.0,-1.0)
    print("\n")

    Q = rng.uniform(low=-1., high=1., size=N)
    Q[:] = 1.0
    np.set_printoptions(precision=4)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    for px in range(N):
        print(60*"-")
        orig = np.zeros(ncomp)
        for lx in range(nterms):
            for mx in range(-1*lx, lx+1):
                py_re = 0.0
                py_im = 0.0

                r = spherical(P[px, :])

                ynm = Yfoo(lx, -1 * mx, r[1], r[2]) * (r[0] ** float(lx))
                ynm *= Q[px]

                py_re += ynm.real
                py_im += ynm.imag

                orig[re_lm(lx, mx)] = py_re
                orig[im_lm(lx, mx)] = py_im
        print(P[px, :], orig)



def neg_p_factor(l,m):
    if m > -1: return 1.0
    m = abs(m)
    return ((-1.0)**m)*(math.factorial(l - m)/math.factorial(l + m))



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


def test_wigner_4_2(phi_set, theta_set):
    """
    Create random rotation matrices, generate moment rotating matrics and check
    they form a function and it's inverse.
    """

    tol = 10.**-14
    jmax = 20

    theta = theta_set
    phi = phi_set

    zr = rotate_z(phi)
    yr = rotate_y(theta)
    rm = np.matmul(yr, zr)

    alpha, beta, gamma = mat2euler(rm, axes='rzyz')

    def mat_norm(a,b):
        return np.linalg.norm(a.ravel() - b.ravel(), np.inf)

    for j in range(0, jmax):

        def err_id(mm):
            return np.linalg.norm(np.eye(2*j+1,2*j+1).ravel() - \
                mm.ravel(), np.inf)

        ii = np.matmul(R_z(j, gamma), R_z(j, -gamma))
        err = err_id(ii)
        if DEBUG:
            print(red_tol(err, tol))
        assert err < tol

        ii = np.matmul(R_y(j, beta), R_y(j, -beta))
        err = err_id(ii)
        if DEBUG:
            print(red_tol(err, tol))
        assert err < tol

        d_1 = np.matmul(R_z(j, -alpha),
                        np.matmul(R_y(j, -beta), R_z(j, -gamma))
                        )
        d_2 = np.matmul(R_z(j, gamma),
                        np.matmul(R_y(j, beta), R_z(j, alpha))
                        )

        err = mat_norm(d_1, R_zyz(p=j,
                                  alpha=-gamma, beta=-beta, gamma=-alpha))
        assert err < tol

        err = mat_norm(d_2, R_zyz(p=j,
                                  alpha=alpha, beta=beta, gamma=gamma))
        assert err < tol

        ii = np.matmul(d_1, d_2)
        err = err_id(ii)
        if DEBUG:
            print(red_tol(err, tol))
        assert err < tol



def test_wigner_4():
    """
    Create random rotation matrices, generate moment rotating matrics and check
    they form a function and it's inverse.
    """

    tol = 10.**-14

    N = 10
    jmax = 20

    atheta = np.random.uniform(low=-2.1*math.pi, high=2.1*math.pi, size=N)
    aphi = np.random.uniform(low=-2.*math.pi, high=2.*math.pi, size=N)

    for ax in range(N):

        theta = atheta[ax]
        phi = aphi[ax]

        zr = rotate_z(phi)
        yr = rotate_y(theta)
        rm = np.matmul(yr, zr)

        alpha, beta, gamma = mat2euler(rm, axes='rzyz')

        def mat_norm(a,b):
            return np.linalg.norm(a.ravel() - b.ravel(), np.inf)

        for j in range(0, jmax):

            def err_id(mm):
                return np.linalg.norm(np.eye(2*j+1,2*j+1).ravel() - \
                    mm.ravel(), np.inf)

            ii = np.matmul(R_z(j, gamma), R_z(j, -gamma))
            err = err_id(ii)
            if DEBUG:
                print(red_tol(err, tol))
            assert err < tol

            ii = np.matmul(R_y(j, beta), R_y(j, -beta))
            err = err_id(ii)
            if DEBUG:
                print(red_tol(err, tol))
            assert err < tol

            d_1 = np.matmul(R_z(j, -alpha),
                            np.matmul(R_y(j, -beta), R_z(j, -gamma))
                            )
            d_2 = np.matmul(R_z(j, gamma),
                            np.matmul(R_y(j, beta), R_z(j, alpha))
                            )

            err = mat_norm(d_1, R_zyz(p=j,
                                      alpha=-gamma, beta=-beta, gamma=-alpha))
            assert err < tol

            err = mat_norm(d_2, R_zyz(p=j,
                                      alpha=alpha, beta=beta, gamma=gamma))
            assert err < tol

            ii = np.matmul(d_1, d_2)
            err = err_id(ii)
            if DEBUG:
                print(red_tol(err, tol))
            assert err < tol







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


def test_wigner_5(phi_set, theta_set):
    """
    rotates random vectors forwards and backwards and checks the results are
    the same. Uses rotate_moments. rotates through two angles
    """

    tol = 10.**-12
    N = 1

    nterms = 40

    ncomp = (nterms**2)*2
    rng = np.random.RandomState(seed=1234)

    P = rng.uniform(low=-1., high=1., size=(N,3))

    #P[0,:] = (-1.0, 1.0, 0.0)

    Q = rng.uniform(low=-1., high=1., size=N)
    Q[:] = 1.0

    orig = np.zeros(ncomp)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    for lx in range(nterms):
        for mx in range(-1*lx, lx+1):
            py_re = 0.0
            py_im = 0.0
            for px in range(N):
                r = spherical(P[px, :])

                ynm = Yfoo(lx, -1 * mx, r[1], r[2]) * (r[0] ** float(lx))
                ynm *= Q[px]

                py_re += ynm.real
                py_im += ynm.imag

            orig[re_lm(lx, mx)] = py_re
            orig[im_lm(lx, mx)] = py_im

    theta = -1.0*math.pi
    phi = -.123*math.pi

    theta = theta_set
    phi = phi_set


    from transforms3d.euler import mat2euler

    zr = rotate_z(phi)
    yr = rotate_y(theta)
    rm = np.matmul(yr, zr)
    irm = np.linalg.inv(rm)

    alpha, beta, gamma = mat2euler(rm, axes='rzyz')

    #alpha, beta, gamma = phi, theta, 0.0


    if DEBUG:
        print("alpha, beta, gamma", alpha, beta, gamma,
              "theta, phi", theta, phi)

    orig_rot = np.zeros(ncomp)
    for lx in range(nterms):
        for mx in range(-1*lx, lx+1):
            py_re = 0.0
            py_im = 0.0
            for px in range(N):

                new_pos = matvec(irm, P[px,:])

                r = spherical(new_pos)
                theta_px = r[1]
                phi_px = r[2]

                ynm = Yfoo(lx, -1 * mx, theta_px, phi_px) * (r[0] ** float(lx))
                ynm *= Q[px]

                py_re += ynm.real
                py_im += ynm.imag

            orig_rot[re_lm(lx, mx)] = py_re
            orig_rot[im_lm(lx, mx)] = py_im

    orig_back_rot = rotate_moments_matrix(
        nterms, alpha=-gamma, beta=-beta, gamma=-alpha, moments=orig_rot)

    err = np.linalg.norm(orig - orig_back_rot, np.inf)

    if DEBUG:
        for nx in range(nterms):
            print("nx =", nx)
            for mx in range(-1*nx, nx+1):
                print("\t{: 2d} | {: .16f} {: .16f} | {: 6f} {: 6f} || {: 6f} {: 6f}".format(mx,
                    orig[re_lm(nx, mx)], orig_back_rot[re_lm(nx, mx)],
                    orig[im_lm(nx, mx)], orig_back_rot[im_lm(nx, mx)],
                    orig_rot[re_lm(nx, mx)], orig_rot[im_lm(nx, mx)]))

        print("ERR:\t", red_tol(err, tol))

    assert err < tol




@pytest.fixture(
    scope="module",
    params=(0.        ,  0.52359878,  1.04719755,  1.57079633,  2.0943951 ,
        2.61799388,  3.14159265 )
)
def theta_set2(request):
    return request.param
def test_wigner_6(theta_set2, phi_set):
#def test_wigner_6():
    """
    rotates random vectors forwards and backwards and checks the results are
    the same. Uses rotate_moments. rotates through two angles
    """

    tol = 10.**-12
    N = 1

    nterms = 5
    ncomp = (nterms**2)*2
    rng = np.random.RandomState(seed=1234)

    P = rng.uniform(low=-1., high=1., size=(N,3))

    P[0,:] = (-1.0, 1.0, 0.0)

    Q = rng.uniform(low=-1., high=1., size=N)
    Q[:] = 1.0

    orig = np.zeros(ncomp)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    for lx in range(nterms):
        for mx in range(-1*lx, lx+1):
            py_re = 0.0
            py_im = 0.0
            for px in range(N):
                r = spherical(P[px, :])

                ynm = Yfoo(lx, -1 * mx, r[1], r[2]) * (r[0] ** float(lx))
                ynm *= Q[px]

                py_re += ynm.real
                py_im += ynm.imag

            orig[re_lm(lx, mx)] = py_re
            orig[im_lm(lx, mx)] = py_im

    radius = 2.0
    theta = theta_set2
    phi = phi_set


    DEBUG = True
    print("\n")

    t0 = time.time()
    orig_mtl = shift_normal(nterms, radius, theta, phi,orig)
    if DEBUG:
        print("standard:\t", time.time() - t0)

    zr = rotate_z(phi)
    yr = rotate_y(theta)
    rm = np.matmul(yr, zr)
    alpha, beta, gamma = mat2euler(rm, axes='rzyz')
    alpha = 0.0
    beta = theta
    gamma = phi

    print("alpha, beta, gamma\t", alpha, beta, gamma)

    t0 = time.time()
    forward_rot = rotate_moments(nterms, alpha=gamma, beta=beta, gamma=alpha,
                                 moments=orig)
    z_mtl = shift_z(nterms, radius, 0.0, forward_rot)
    rot_mtl = rotate_moments(nterms, alpha=-alpha, beta=-beta, gamma=-gamma,
                             moments=z_mtl)
    if DEBUG:
        print("rotated:\t", time.time() - t0)

    err = np.linalg.norm(rot_mtl - orig_mtl, np.inf)

    if DEBUG and False:
        for nx in range(nterms):
            print("nx =", nx)
            for mx in range(-1*nx, nx+1):
                print("\t{: 2d} | {: 6f} {: 6f} | {: 6f} {: 6f}".format(mx,
                    orig_mtl[re_lm(nx, mx)], rot_mtl[re_lm(nx, mx)],
                    orig_mtl[im_lm(nx, mx)], rot_mtl[im_lm(nx, mx)]))

    if DEBUG:
        print("ERR:\t", red_tol(err, tol))



def test_wigner_6_pi():
    """
    rotates random vectors forwards and backwards and checks the results are
    the same. Uses rotate_moments. rotates through two angles
    """

    tol = 10.**-12
    N = 1

    nterms = 5

    ncomp = (nterms**2)*2
    rng = np.random.RandomState(seed=1234)

    P = rng.uniform(low=-1., high=1., size=(N,3))

    P[0,:] = (-1.0, 1.0, 0.0)

    Q = rng.uniform(low=-1., high=1., size=N)
    Q[:] = 1.0

    orig = np.zeros(ncomp)

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + nterms**2

    for lx in range(nterms):
        for mx in range(-1*lx, lx+1):
            py_re = 0.0
            py_im = 0.0
            for px in range(N):
                r = spherical(P[px, :])

                ynm = Yfoo(lx, -1 * mx, r[1], r[2]) * (r[0] ** float(lx))
                ynm *= Q[px]

                py_re += ynm.real
                py_im += ynm.imag

            orig[re_lm(lx, mx)] = py_re
            orig[im_lm(lx, mx)] = py_im



    radius = 2.0
    theta = 0.0
    theta = 3.14159265
    phi = 1.23


    DEBUG = True
    print("\n")

    t0 = time.time()
    orig_mtl = shift_normal(nterms, radius, theta, phi,orig)
    if DEBUG:
        print("standard:\t", time.time() - t0)

    zr = rotate_z(phi)
    yr = rotate_y(theta)
    rm = np.matmul(yr, zr)

    alpha, beta, gamma = mat2euler(rm, axes='rzyz')
    #alpha = 0.0
    #beta = -theta
    #gamma = 0.0

    print("alpha, beta, gamma\t", alpha, beta, gamma)

    t0 = time.time()
    forward_rot = rotate_moments(nterms, alpha=gamma, beta=beta, gamma=alpha,
                                 moments=orig)

    z_mtl = shift_z(nterms, radius, 0.0, forward_rot)

    rot_mtl = rotate_moments(nterms, alpha=-alpha, beta=-beta, gamma=-gamma,
                             moments=z_mtl)

    if DEBUG:
        print("rotated:\t", time.time() - t0)

    err = np.linalg.norm(rot_mtl - orig_mtl, np.inf)

    if DEBUG and False:
        for nx in range(nterms):
            print("nx =", nx)
            for mx in range(-1*nx, nx+1):
                print("\t{: 2d} | {: 6f} {: 6f} | {: 6f} {: 6f}".format(mx,
                    orig_mtl[re_lm(nx, mx)], rot_mtl[re_lm(nx, mx)],
                    orig_mtl[im_lm(nx, mx)], rot_mtl[im_lm(nx, mx)]))

    if DEBUG:
        print("ERR:\t", red_tol(err, tol))





def test_wigner_c_1():
    tol = 10.**-14
    beta = 0.566
    jmax = 50

    E = wigner._WignerEngine()

    pointers, matrices = E(jmax, beta)

    for jx in range(1,jmax):
        if DEBUG:
            print(jx)
        for mpx in range(-jx, jx+1):
            if DEBUG:
                print("\t", mpx)
            for mx in range(-jx, jx+1):
                aa = wigner_d_rec(jx, mpx, mx, beta)
                bb = matrices[jx][mpx+jx, mx+jx]

                err = abs(aa - bb)
                if err > tol:
                    serr = red(err)
                else:
                    serr = green(err)

                if DEBUG:
                    print("\t\t{: .8f} {: .8f}".format(aa, bb),
                          "\t", serr)
                assert err < tol*jx


def test_wigner_c_2():
    tol = 10.**-14
    jmax = 20

    E = wigner._WignerEngine()

    for beta in np.random.uniform(low=-2.*math.pi, high=2.0*math.pi, size=40):

        pointers, matrices = E(jmax, beta)

        for jx in range(1,jmax):
            if DEBUG:
                print(jx)
            for mpx in range(-jx, jx+1):
                if DEBUG:
                    print("\t", mpx)
                for mx in range(-jx, jx+1):
                    aa = wigner_d_rec(jx, mpx, mx, beta)
                    bb = matrices[jx][mpx+jx, mx+jx]

                    err = abs(aa - bb)
                    if err > tol:
                        serr = red(err)
                    else:
                        serr = green(err)

                    if DEBUG:
                        print("\t\t{: .8f} {: .8f}".format(aa, bb),
                              "\t", serr)
                    assert err < tol*jx





def test_wigner_c_3():
    tol = 10.**-15
    jmax = 20
    alpha = 0.0
    gamma = 0.1
    beta=0.45745

    opr, opi, om = Rzyz_set_orig(jmax, alpha, beta, gamma, ctypes.c_double)
    npr, npi, nm= Rzyz_set(jmax, alpha, beta, gamma, ctypes.c_double)

    DEBUG = False
    for jx in range(jmax):
        if DEBUG:
            print(jx)
        for mpx in range(-jx, jx+1):
            if DEBUG:
                print("\t", mpx)
            for mx in range(-jx, jx+1):
                aa = om['real'][jx][mpx, mx]
                bb = nm['real'][jx][mpx, mx]

                err = abs(aa - bb)
                if err > tol:
                    serr = red(err)
                else:
                    serr = green(err)

                if DEBUG:
                    print("\t\t{: .8f} {: .8f}".format(aa, bb),
                          "\t", serr)

                assert err < tol





