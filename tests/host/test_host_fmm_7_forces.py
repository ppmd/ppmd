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

cube_offsets = (
    (-1,1,-1),
    (-1,-1,-1),
    (-1,0,-1),
    (0,1,-1),
    (0,-1,-1),
    (0,0,-1),
    (1,0,-1),
    (1,1,-1),
    (1,-1,-1),

    (-1,1,0),
    (-1,0,0),
    (-1,-1,0),
    (0,-1,0),
    (0,1,0),
    (1,0,0),
    (1,1,0),
    (1,-1,0),

    (-1,0,1),
    (-1,1,1),
    (-1,-1,1),
    (0,0,1),
    (0,1,1),
    (0,-1,1),
    (1,0,1),
    (1,1,1),
    (1,-1,1)
)

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
    #print("mx: {: >30} , exp: {: >30} P: {: >30} coeff: {: >30}".format(
    #    mx,cmath.exp(1.j * mx * phi), legp.real, coeff))
    
    return coeff * legp * cmath.exp(1.j * mx * phi)


def compute_phi(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):
            #print('mx', mx)

            re_exp = np.cos(mx*disp_sph[0,2])
            im_exp = np.sin(mx*disp_sph[0,2])

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = 1. / (disp_sph[0,0] ** (lx+1.))

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

    return phi_sph_re, phi_sph_im



def compute_phi_local(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(mx*disp_sph[0,2])
            im_exp = np.sin(mx*disp_sph[0,2])

            #print('mx', mx, im_exp)

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = disp_sph[0,0] ** (lx)

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

    return phi_sph_re, phi_sph_im


def get_p_exp(fmm, disp_sph):
    def re_lm(l,m): return (l**2) + l + m
    exp_array = np.zeros(fmm.L*8 + 2, dtype=ctypes.c_double)
    p_array = np.zeros((fmm.L*2)**2, dtype=ctypes.c_double)
    for lx in range(fmm.L*2):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        for mxi, mx in enumerate(mrange2):
            coeff = math.sqrt(float(math.factorial(lx-abs(mx)))/
                math.factorial(lx+abs(mx)))
            p_array[re_lm(lx, mx)] = scipy_p[mxi].real*coeff

    for mxi, mx in enumerate(list(
            range(-2*fmm.L, 1)) + list(range(1, 2*fmm.L+1))
        ):

        exp_array[mxi] = np.cos(mx*disp_sph[0,2])
        exp_array[mxi + fmm.L*4 + 1] = np.sin(mx*disp_sph[0,2])

    return p_array, exp_array


def test_fmm_oct_1():

    R = 3
    eps = 10.**-2
    free_space = True

    N = 2
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    ASYNC = False
    DIRECT = True if MPISIZE == 1 else False

    DIRECT= True

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=1234)



    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        A.P[0,:] = (nra, nra, 0)
        A.P[1,:] = (nra, ra, 0)
        A.P[2,:] = (ra, nra, 0)
        A.P[3,:] = (ra, ra, 0)

        A.Q[0,0] = -1.
        A.Q[3,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = 1.

    elif N == 1:
        A.P[0,:] = ( 0.25*E, 0.25*E, 0.25*E)
        A.P[0,:] = ( 10.**-6, 10.**-6, 10.**-6)

        #A.P[0,:] = (0, -0.25*E, 0)
        #A.P[1,:] = (0, 0.25*E, 0)
        #A.P[0,:] = (0, 0, -0.25*E)
        #A.P[1,:] = (0, 0, 0.25*E)

        #A.Q[:,0] = 1.

        A.Q[0,0] = 1.

    elif N == 2:
        #A.P[0,:] = ( 0.25*E, 0.25*E, 0.25*E)
        #A.P[1,:] = ( -0.25*E, -0.25*E, 0)

        #A.P[0,:] = (0, -0.25*E, 0)
        #A.P[1,:] = (0, 0.25*E, 0)
        #A.P[0,:] = (0, 0, -0.25*E)
        #A.P[1,:] = (0, 0, 0.25*E)

        #A.Q[:,0] = 1.
        eps = 0.00001

        epsx = 0.0
        epsy = eps
        epsz = 0.0

        A.P[0] = ( 0.5 + epsx, 1.5 + epsy, 0.5 + epsz)
        A.P[1] = ( 0.5 + epsx,-0.5 + epsy, 0.5 + epsz)

        A.Q[0,0] = 1.
        A.Q[1,0] = -1.

    elif N == 8:
        for px in range(8):
            phi = (float(px)/8) * 2. * math.pi
            pxr = 0.25*E
            pxx = pxr * math.cos(phi)
            pxy = pxr * math.sin(phi)


            A.P[px, :] = (pxx, pxy, 0)
            A.Q[px, 0] = 1. - 2. * (px % 2)
            #A.Q[px, 0] = -1.

        #A.P[0,:] += eps


        eps = 0.00001
        #A.P[0:N:2,0] += eps
        #A.P[0,0] -= eps
        A.P[4,0] -= eps
        A.P[:, 2] -= 0.200
        A.P[:, 1] -= 0.200

        #A.Q[0,0] = 0.
        A.Q[1,0] = 0.
        A.Q[4,0] = 0.
        A.Q[3,0] = 0.
        A.Q[5,0] = 0.
        A.Q[6,0] = 0.
        A.Q[7,0] = 0.


    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, async=ASYNC)
    phi_py = fmm(A.P, A.Q, forces=A.F, async=ASYNC)
    t1 = time.time()


    for px in range(N):
        line = ('{: 8.4f} | {: 8.4f} {: 8.4f} {: 8.4f} | ' + \
                '{: 8.4f} {: 8.4f} {: 8.4f}').format(
            A.Q[px, 0],
            A.P[px,0], A.P[px, 1], A.P[px, 2],
            A.F[px,0], A.F[px, 1], A.F[px, 2]
        )

        print(line)


    print("MID")
    for lx in range(2):
        print("lx", lx)
        for mx in range(-1*lx, lx+1):
            print("\tmx", mx, "\t:", fmm.up[fmm.re_lm(lx, mx)],
                  fmm.up[fmm.im_lm(lx, mx)])
    print("PX=0")
    for lx in range(2):
        print("lx", lx)
        for mx in range(-1*lx, lx+1):
            print("\tmx", mx, "\t:", fmm.tree_plain[2][2,3,2,fmm.re_lm(lx, mx)],
                  fmm.tree_plain[2][2,3,2,fmm.im_lm(lx, mx)])





    for lx in range(fmm.L):
        for mx in range(-1*lx, lx+1):
            py_re = 0.0
            py_im = 0.0
            for px in range(N):
                r = spherical(A.P[px, :])
                ynm = Yfoo(lx, -1 * mx, r[1], r[2]) * (r[0] ** float(lx)) * A.Q[px, 0]
                py_re += ynm.real
                py_im += ynm.imag

            assert abs(py_re - fmm.up[fmm.re_lm(lx, mx)]) < 10**-10
            assert abs(py_im - fmm.up[fmm.im_lm(lx, mx)]) < 10**-10

            #if DEBUG:
            #    print("\t{: >5} | {: >30} {: >30} | {: >30} {: >30} ".format(
            #        mx, py_re, fmm.up[fmm.re_lm(lx, mx)],
            #        py_im, fmm.up[fmm.im_lm(lx, mx)]))


    if DIRECT:
        phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
            if free_space == '27':
                for ofx in cube_offsets:


                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
    else:
        if free_space == '27':
            phi_direct = -0.12868996439494947981
        elif free_space == True:
            phi_direct = -0.12131955438932764957
        else:
            raise RuntimeError("bad parameter")



    local_err = abs(phi_py - phi_direct)
    if local_err > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print("\n")
        #print(60*"-")
        #opt.print_profile()
        #print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("ENERGY DIRECT:\t{:.20f}".format(phi_direct))
        print("ENERGY FMM:\t", phi_py)
        print("ERR:\t\t", serr)

    cell_mids = ((0.5, 1.5, 0.5),(0.5,-0.5,0.5))
    cell_ind = ((2,3,2), (2,1,2))


    for px in range(2):
        print(green(60*'-'))
        dx = A.P[px, :] - np.array(cell_mids[px])

        sph = spherical(dx)
        radius = sph[0]
        theta = sph[1]
        phi = sph[2]
        
        print(px, radius, phi, theta)

        rstheta = 0.0
        if abs(theta) > 0.0:
            rstheta = 1./sin(theta)


        rhat = np.array((cos(phi)*sin(theta),
                        sin(phi)*sin(theta),
                        cos(theta)))

        thetahat = np.array(
            (cos(phi)*cos(theta), sin(phi)*cos(theta), -1.0 * sin(theta))
        )

        phihat = np.array((-1*sin(phi), cos(phi), 0.0))

        Fv = np.zeros(3)

        plain = fmm.tree_plain[fmm.R-1][
                           cell_ind[px][2],
                           cell_ind[px][1],
                           cell_ind[px][0],
                           :]

        for jx in range(2):
            print(green(jx))
            for kx in range(-1*jx, jx+1):
                print("\t", red(kx))
                rpower = radius**(jx-1.)
                Ljk = plain[fmm.re_lm(jx,kx)] + 1.j*plain[fmm.re_lm(jx,kx)]

                # radius
                radius_coeff = float(jx) * rpower * \
                      Yfoo(jx, kx, theta, phi)

                # theta
                theta_coeff = float(jx - abs(kx) + 1) * \
                                Pfoo(jx+1, abs(kx), cos(theta))
                theta_coeff -= float(jx + 1) * cos(theta) * \
                                Pfoo(jx, abs(kx), cos(theta))
                theta_coeff *= rpower * rstheta

                # phi
                phi_coeff = Yfoo(jx, kx, theta, phi) * (1.j * float(kx))
                phi_coeff *= rpower * rstheta

                Fv -= rhat *     (Ljk* radius_coeff).real +\
                      thetahat * (Ljk* theta_coeff ).real +\
                      phihat *   (Ljk* phi_coeff   ).real

                print("{: >8} {: >60} | {: >8} {: >60} | {: >8} {: >60}".format(
                      yellow("r"), radius_coeff,
                      yellow("theta"),theta_coeff,
                      yellow("phi"), phi_coeff))

                print("{} = {} * {}".format(
                    Ljk* radius_coeff, radius_coeff, Ljk
                ))


        print(px, Fv)




    #assert local_err < eps


















