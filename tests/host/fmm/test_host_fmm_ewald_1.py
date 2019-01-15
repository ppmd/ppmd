__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../../res'), filename)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

directions= (
    ((-0.25, 0, 0), (0.25, 0, 0)),
    ((0, -0.25, 0), (0, 0.25, 0)),
    ((0, 0, -0.25), (0, 0, 0.25)),
)

charges = (1, -1)

@pytest.mark.parametrize("direction", directions)
@pytest.mark.parametrize("charge", charges)
def test_fmm_ewald_1(direction, charge):

    R = 3
    L = 12

    E = 10.

    N = 2
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    
    A.P[0,:] = direction[0]
    A.P[1,:] = direction[1]
    A.P[:1, :] *= E

    A.P[:1, 0] -= 0.2

    A.Q[0,0] =  1.0 * charge
    A.Q[1,0] = -1.0 * charge

    A.scatter_data_from(0)


    lr_fmm = PyFMM(domain=A.domain, r=R, l=L, free_space=False)
    lr_ewald = EwaldOrthoganalHalf(A.domain, real_cutoff=rc, eps=10.**-8)

    phi_fmm = lr_fmm(A.P, A.Q)
    phi_ewald = lr_ewald(A.P, A.Q, A.F)

    diff = abs(phi_fmm - phi_ewald)
    assert diff < 10.**-4
    

