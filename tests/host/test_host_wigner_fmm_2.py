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
from ppmd.coulomb.wigner import _wigner_engine
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


def test_new_matvec_1():
    R = 3
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

    fmm = PyFMM(domain=A.domain, r=R, l=L, free_space=free_space
                , _debug=True)

    for alpha, beta in itertools.product(angle_set, angle_set):

        exp_re = np.zeros(L-1, dtype=REAL)
        exp_im = np.zeros(L-1, dtype=REAL)
        for mxi, mx in enumerate(range(1, L)):
            me = cmath.exp(1.j * mx * alpha)
            exp_re[mxi] = me.real
            exp_im[mxi] = me.imag

        wp, wm = _wigner_engine(L, beta, eps_scaled=True)

        for lx in range(L):

            nc = 2*lx+1
            re_x = np.zeros(nc, dtype=REAL)
            im_x = np.zeros(nc, dtype=REAL)
            re_x[:] = rng.uniform(size=nc)
            im_x[:] = rng.uniform(size=nc)
            re_bz = np.zeros(nc, dtype=REAL)
            im_bz = np.zeros(nc, dtype=REAL)
            re_by = np.zeros(nc, dtype=REAL)
            im_by = np.zeros(nc, dtype=REAL)

            lib_forw = fmm._translate_mtlz2_lib['wrapper_rotate_p_forward']

            lib_forw(
                INT32(lx),
                exp_re.ctypes.get_as_parameter(),
                exp_im.ctypes.get_as_parameter(),
                ctypes.c_void_p(int(wp[lx])),
                re_x.ctypes.get_as_parameter(),
                im_x.ctypes.get_as_parameter(),
                re_bz.ctypes.get_as_parameter(),
                im_bz.ctypes.get_as_parameter(),
                re_by.ctypes.get_as_parameter(),
                im_by.ctypes.get_as_parameter()
            )

            x = np.zeros(nc, dtype=np.complex)
            x.real[:] = re_x[:]
            x.imag[:] = im_x[:]

            o0 = np.zeros(nc, dtype=np.complex)
            o0.real[:] = re_bz[:]
            o0.imag[:] = im_bz[:]

            o1 = np.zeros(nc, dtype=np.complex)
            o1.real[:] = re_by[:]
            o1.imag[:] = im_by[:]

            A_alpha = np.zeros((nc,nc), dtype=np.complex)

            for mxi, mx, in enumerate(range(-lx, lx+1)):
                A_alpha[mxi, mxi] = cmath.exp(1.j * mx * alpha)

            A_a_x = np.dot(A_alpha, x)

            err0 = np.linalg.norm(A_a_x - o0, np.inf)
            assert err0 < (10.**-13)

            Ab_Aa_x = np.dot(wm[lx], A_a_x)

            err1 = np.linalg.norm(Ab_Aa_x - o1, np.inf)
            assert err1 < (10.**-13)











