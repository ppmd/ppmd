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


def test_fmm_force_ewald_1():

    R = 3
    eps = 10.**-6
    free_space = False

    N = 4
    E = 4.
    rc = E/8

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.FE = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)


    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        A.P[0,:] = ( 0.21,  0.21, 0.00)
        A.P[1,:] = (-0.21,  0.21, 0.00)
        A.P[2,:] = (-0.21, -0.21, 0.00)
        A.P[3,:] = ( 0.21, -0.21, 0.00)

        #A.P[0,:] = ( 0.00,  0.21, 0.21)
        #A.P[1,:] = ( 0.00,  0.21,-0.21)
        #A.P[2,:] = ( 0.00, -0.21,-0.21)
        #A.P[3,:] = ( 0.00, -0.21, 0.21)

        #A.P[0,:] = ( 1.01,  1.01, 0.00)
        #A.P[1,:] = (-1.01,  1.01, 0.00)
        #A.P[2,:] = (-1.01, -1.01, 0.00)
        #A.P[3,:] = ( 1.01, -1.01, 0.00)

        A.Q[0,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = -1.
        A.Q[3,0] = 1.

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
        ra = 0.25 * E
        nra = -0.25 * E

        eps = 0.00

        epsx = 0
        epsy = 0
        epsz = 0

        A.P[0,:] = ( 0.001, 0.001,  1.001)
        A.P[1,:] = ( 0.001, 0.001, -1.001)

        #A.P[:2:,:] = rng.uniform(low=-0.4999*E, high=0.4999*E, size=(N,3))

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

    else:
        assert N % 2 == 0
        for px in range(N//2):
            pos = rng.uniform(low=-0.4999*E, high=0.4999*E, size=(1,3))
            cha = rng.uniform(low=-1., high=1.)

            A.P[px, :] = pos
            A.Q[px, 0] = cha

            A.P[-1*(px+1), :] = -1.0*pos
            A.Q[-1*(px+1), 0] = cha

        bias = np.sum(A.Q[:])
        A.Q[:,0] -= bias/N

    dipole = np.zeros(3)
    for px in range(N):
        dipole[:] += A.P[px,:]*A.Q[px,0]

    bias = np.sum(A.Q[:])

    print("DIPOLE:\t", dipole, "TOTAL CHARGE:\t", bias)

    A.scatter_data_from(0)

    ewald = EwaldOrthoganalHalf(
        domain=A.domain,
        real_cutoff=rc,
        eps=10.**-12,
        shared_memory=SHARED_MEMORY
    )

    t2 = time.time()
    ewald.evaluate_contributions(positions=A.P, charges=A.Q)
    A.cri[0] = 0.0
    ewald.extract_forces_energy_reciprocal(A.P, A.Q, A.FE, A.cri)
    A.crr[0] = 0.0
    ewald.extract_forces_energy_real(A.P, A.Q, A.FE, A.crr)
    A.crs[0] = 0.0
    ewald.evaluate_self_interactions(A.Q, A.crs)

    t3 = time.time()

    phi_ewald = A.cri[0] + A.crr[0] + A.crs[0]


    for px in range(N):

        print("PX:", px)
        print("\t\tFORCE EWALD :",A.FE[px,:])





