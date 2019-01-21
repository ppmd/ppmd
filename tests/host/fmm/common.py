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

