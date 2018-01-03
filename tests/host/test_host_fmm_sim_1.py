from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'),
                        filename)


np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools


from ppmd import *
from ppmd.coulomb.fmm import *
from scipy.special import sph_harm, lpmv
import time


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True

def test_fmm_sim_1():

    input_data = np.load(get_res_file_path('coulomb/CO2cuboid.npy'))
    N = input_data.shape[0]

    E = 50.

    A = state.State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-6

    fmm = PyFMM(domain=A.domain, r=3, N=N, eps=eps, free_space=True)

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    if MPIRANK == 0:
        A.P[:] = input_data[:,0:3:]
        A.Q[:, 0] = input_data[:,3]
    A.scatter_data_from(0)

    if MPISIZE == 1:
        # compute potential energy to point across all charges directly
        P2 = data.PositionDat(npart=N, ncomp=3)
        Q2 = data.ParticleDat(npart=N, ncomp=1)
        P2[:,:] = A.P[:N:,:]
        Q2[:,:] = A.Q[:N:,:]
        phi_ga = data.ScalarArray(ncomp=1, dtype=ctypes.c_double)
        src = """
        const double d0 = P.j[0] - P.i[0];
        const double d1 = P.j[1] - P.i[1];
        const double d2 = P.j[2] - P.i[2];
        phi[0] += 0.5 * Q.i[0] * Q.j[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
        """
        phi_kernel = kernel.Kernel('all_to_all_phi', src,
                                   headers=(kernel.Header('math.h'),))


        phi_loop = pairloop.AllToAllNS(kernel=phi_kernel,
                                       dat_dict={'P': P2(access.READ),
                                                 'Q': Q2(access.READ),
                                                 'phi': phi_ga(access.INC_ZERO)})
        phi_loop.execute()
        phi_direct = phi_ga[0]
    else:
        phi_direct = None

    t0 = time.time()
    phi_fmm = fmm(A.P, A.Q)
    t1 = time.time()

    if DEBUG and MPIRANK == 0:
        print("Time:", t1 - t0)
        print("MPISIZE", MPISIZE)
        print("phi_fmm", phi_fmm)
        if MPISIZE == 1:
            print("phi_direct: {:.30f}".format(phi_direct))
            print("phi_fmm", phi_fmm)
            print("ERROR:", abs(phi_direct - phi_fmm))





