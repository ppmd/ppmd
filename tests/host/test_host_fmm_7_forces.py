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

def red_tol(val, tol):
    if abs(val) > tol:
        return red(str(val))
    else:
        return green(str(val))


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
    eps = 10.**-4
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
        eps = 0.00

        epsx = 0
        epsy = 0
        epsz = 0

        A.P[0] = ( -0.5+eps, -0.5+eps,  -0.5+eps)
        A.P[1] = (  1.5+eps,  1.5+eps,  1.5+eps)

        A.P[:2:,:] = rng.uniform(low=-0.4999*E, high=0.4999*E, size=(N,3))

        A.Q[0,0] = -1.
        A.Q[1,0] = 1.

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

    print("PX=1")
    for lx in range(3):
        print("lx", lx)
        for mx in range(-1*lx, lx+1):
            print("\tmx", mx, "\t:", fmm.tree_plain[2][2,2,2,fmm.re_lm(lx, mx)],
                  fmm.tree_plain[2][2,2,2,fmm.im_lm(lx, mx)])



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

    direct_forces = np.zeros((N, 3))

    if DIRECT:
        #print("WARNING 0-th PARTICLE ONLY")
        phi_direct = 0.0

        # compute phi from image and surrounding 26 cells

        for ix in range(N):


            phi_part = 0.0
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
                phi_part += A.Q[ix, 0] * A.Q[jx, 0] /rij

                direct_forces[ix,:] -= A.Q[ix, 0] * A.Q[jx, 0] * \
                                       (A.P[jx,:] - A.P[ix,:]) / (rij**3.)
                direct_forces[jx,:] += A.Q[ix, 0] * A.Q[jx, 0] * \
                                       (A.P[jx,:] - A.P[ix,:]) / (rij**3.)
            if free_space == '27':
                for ofx in cube_offsets:
                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
                        phi_part += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij

                        direct_forces[ix,:] -= A.Q[ix, 0] * A.Q[jx, 0] * \
                                           (A.P[jx,:] - A.P[ix,:] + cube_mid) \
                                               / (rij**3.)


            print("ix:", ix, "phi:", phi_part)

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

    h = 0.001
    stencil = (
        np.array(( h, 0.0   , 0.0   )),
        np.array(( -1.*h, 0.0   , 0.0   )),
        np.array(( 0.0  , h , 0.0   )),
        np.array(( 0.0  , -1.*h , 0.0   )),
        np.array(( 0.0  , 0.0   ,h )),
        np.array(( 0.0  , 0.0   ,-1.*h )),
    )


    phi_force = 0.0
    for px in range(N):
        print(green(60*'-'))

        cell = np.array((
            int((A.P[px, 0]+E*0.5)),
            int((A.P[px, 1]+E*0.5)),
            int((A.P[px, 2]+E*0.5))
        ))

        width = E/(2.**(R-1))
        start = (E*-0.5) + 0.5*width

        cell_mid = np.array((
            start + cell[0]*width,
            start + cell[1]*width,
            start + cell[2]*width
        ))

        dx = A.P[px, :] - cell_mid

        sph = spherical(dx)
        radius = sph[0]
        #radius = 0.000000001
        theta = sph[1]
        phi = sph[2]
        
        print(px, radius, phi, theta)

        rstheta = 0.0
        if abs(sin(theta)) > 0.0:
            rstheta = 1./sin(theta)

        print("1./sin(theta)", rstheta)

        rhat = np.array((cos(phi)*sin(theta),
                        sin(phi)*sin(theta),
                        cos(theta)))

        thetahat = np.array(
            (cos(phi)*cos(theta)*rstheta,
             sin(phi)*cos(theta)*rstheta,
             -1.0*sin(theta)*rstheta)
        )

        print("sin(theta)/sin(theta)", sin(theta)*rstheta,
              "cos(theta)", cos(theta))

        phihat = np.array((-1*sin(phi), cos(phi), 0.0))

        Fv = np.zeros(3)
        Fvs = np.zeros(3)
        Fvs_im = np.zeros(3)

        vec = phihat
        print("vectors:\t", rhat, phihat, thetahat, np.linalg.norm(vec))

        plain = fmm.tree_plain[R-1][cell[2], cell[1], cell[0], :]
        phi_part = 0.0

        for jx in range(0,fmm.L):
            #print(green(jx))
            for kx in range(-1*jx, jx+1):
                #print("\t", red(kx))

                Ljk = plain[fmm.re_lm(jx,kx)] + 1.j*plain[fmm.im_lm(jx,kx)]
                mid_phi = A.Q[px, 0]*(radius**jx)* \
                             Ljk*Yfoo(jx,kx,theta, phi)

                # energy
                phi_force += 0.5 * mid_phi
                phi_part += 0.5 * mid_phi

                # Force from finite differences
                # x
                pos = spherical(dx + stencil[0])
                phip = A.Q[px, 0]*(pos[0]**jx)* Ljk*Yfoo(jx,kx,pos[1], pos[2])
                pos = spherical(dx + stencil[1])
                phin = A.Q[px, 0]*(pos[0]**jx)* Ljk*Yfoo(jx,kx,pos[1], pos[2])
                Fvs[0] -= (phip - phin).real/(2.*h)
                Fvs_im[0] -= (phip - phin).imag/(2.*h)
                # y
                pos = spherical(dx + stencil[2])
                phip = A.Q[px, 0]*(pos[0]**jx)* Ljk*Yfoo(jx,kx,pos[1], pos[2])
                pos = spherical(dx + stencil[3])
                phin = A.Q[px, 0]*(pos[0]**jx)* Ljk*Yfoo(jx,kx,pos[1], pos[2])
                Fvs[1] -= (phip - phin).real/(2.*h)
                Fvs_im[1] -= (phip - phin).imag/(2.*h)
                # z
                pos = spherical(dx + stencil[4])
                phip = A.Q[px, 0]*(pos[0]**jx)* Ljk*Yfoo(jx,kx,pos[1], pos[2])
                pos = spherical(dx + stencil[5])
                phin = A.Q[px, 0]*(pos[0]**jx)* Ljk*Yfoo(jx,kx,pos[1], pos[2])
                Fvs[2] -= (phip - phin).real/(2.*h)
                Fvs_im[2] -= (phip - phin).imag/(2.*h)


                # force from gradiant of sperical harmonics
                if jx == 1:
                    rpower = 1.0
                else:
                    if radius > 0:
                        rpower = radius**(jx-1.)
                    else:
                        rpower = 0.0

                # radius
                #radius_coeff = 0.0
                radius_coeff = float(jx) * rpower * \
                                              Yfoo(jx, kx, theta, phi)

                # theta
                theta_coeff = float(jx - abs(kx) + 1) * \
                                Pfoo(jx+1, abs(kx), cos(theta))
                theta_coeff -= float(jx + 1) * cos(theta) * \
                                Pfoo(jx, abs(kx), cos(theta))
                theta_coeff *= rpower
                theta_coeff *= Hfoo(jx, kx) * cmath.exp(1.j * float(kx) * phi)

                # phi
                phi_coeff = Yfoo(jx, kx, theta, phi) * (1.j * float(kx))
                phi_coeff *= rpower * rstheta

                #radius_coeff = 0.0
                #theta_coeff = 0.0
                #phi_coeff = 0.0

                Fv -= A.Q[px, 0] * (rhat     * (Ljk* radius_coeff).real +\
                                    thetahat * (Ljk* theta_coeff ).real +\
                                    phihat   * (Ljk* phi_coeff   ).real)

                ntol = 0.001
                pbool = False
                if (abs(radius_coeff) > ntol or abs(theta_coeff) > ntol or \
                    abs(phi_coeff) > ntol) and pbool:
                    print(jx, kx)
                    print("{: >8} {: >60} | {: >8} {: >60} | {: >8} {: >60}".format(

                        yellow("r"), radius_coeff,
                          yellow("theta"),theta_coeff,
                          yellow("phi"), phi_coeff))

                continue
                print("{} = {} * {}".format(
                    Ljk* radius_coeff, radius_coeff, Ljk
                ))


        err_re_f = red_tol(np.linalg.norm(direct_forces[px,:] - Fv, ord=np.inf), 10.**-6)
        err_re_c = red_tol(np.linalg.norm(direct_forces[px,:] - A.F[px,:], ord=np.inf), 10.**-6)
        err_re_s = red_tol(np.linalg.norm(direct_forces[px,:] - Fvs, ord=np.inf), 10.**-6)
        err_im_s = red_tol(np.linalg.norm(Fvs_im), 10.**-6)

        print("PX:", px)
        print("\t\tFORCE FMM :",Fv, err_re_f)
        print("\t\tFORCE FMMC:",A.F[px,:], err_re_c)
        print("\t\tAPPRX REAL:",Fvs, err_re_s)
        print("\t\tAPPRX IMAG:",Fvs_im, err_im_s)


        print("PHI_PART", px, phi_part)

    print("Energy again:\t", phi_force, "ERR:\t", red(abs(phi_force.real-phi_py)))



    other = ( 0.5,-0.5, 0.5)
    point = np.array((0.5, 1.2, 0.5))

    dphi = 1./np.linalg.norm(point - other)

    cell = np.array((
        int((point[0]+2)),
        int((point[1]+2)),
        int((point[2]+2))
    ))

    mid = np.array((-1.5, -1.5, -1.5)) + cell

    sph = spherical(point - mid)
    plain = fmm.tree_plain[fmm.R-1][ cell[2], cell[1], cell[0], :]
    point_phi = 0.0
    for jx in range(fmm.L):
        for kx in range(-1*jx, jx+1):
            Ljk = plain[fmm.re_lm(jx,kx)] + 1.j*plain[fmm.re_lm(jx,kx)]

            point_phi += (sph[0]**jx)* Ljk*Yfoo(jx,kx,sph[1], sph[2])

    print("POINT PHI:\t", point_phi.real, sph, cell)
    print("OTHER PHI:\t", dphi, "DX:", point-other)




    point = np.array(( 0.5,-0.2, 0.5))
    other = (0.5, 1.5, 0.5)

    dphi = 1./np.linalg.norm(point - other)

    cell = np.array((
        int((point[0]+2)),
        int((point[1]+2)),
        int((point[2]+2))
    ))

    mid = np.array((-1.5, -1.5, -1.5)) + cell

    sph = spherical(point - mid)
    plain = fmm.tree_plain[fmm.R-1][ cell[2], cell[1], cell[0], :]
    point_phi = 0.0
    for jx in range(fmm.L):
        for kx in range(-1*jx, jx+1):
            Ljk = plain[fmm.re_lm(jx,kx)] + 1.j*plain[fmm.re_lm(jx,kx)]

            point_phi += (sph[0]**jx)* Ljk*Yfoo(jx,kx,sph[1], sph[2])

    print("POINT PHI:\t", point_phi.real, sph, cell)
    print("OTHER PHI:\t", dphi, "DX:", point-other)













