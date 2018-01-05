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



vv_kernel1_code = '''
const double M_tmp = 1.0/M.i[0];
V.i[0] += dht*F.i[0]*M_tmp;
V.i[1] += dht*F.i[1]*M_tmp;
V.i[2] += dht*F.i[2]*M_tmp;
P.i[0] += dt*V.i[0];
P.i[1] += dt*V.i[1];
P.i[2] += dt*V.i[2];
'''

vv_kernel2_code = '''
const double M_tmp = 1.0/M.i[0];
V.i[0] += dht*F.i[0]*M_tmp;
V.i[1] += dht*F.i[1]*M_tmp;
V.i[2] += dht*F.i[2]*M_tmp;
k[0] += (V.i[0]*V.i[0] + V.i[1]*V.i[1] + V.i[2]*V.i[2])*0.5*M.i[0];
'''

PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
GlobalArray = data.GlobalArray
State = state.State
ParticleLoop = loop.ParticleLoopOMP
Pairloop = pairloop.PairLoopNeighbourListNSOMP
PBC = domain.BoundaryTypePeriodic


def test_fmm_sim_1():

    R = 3
    eps = 10.**-6
    free_space = False

    dt = 0.001
    shell_steps = 10
    steps = 400

    crn = 10
    rho = 3.

    N = int(crn**3)
    E = rho * crn

    lattice = True


    if MPIRANK == 0:
        print("N:\t", N)

    A = State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = PBC()

    CUDA=True

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda=CUDA)

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.V = data.ParticleDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.M = data.ParticleDat(ncomp=1)
    A.Q = data.ParticleDat(ncomp=1)

    A.u = data.GlobalArray(ncomp=1)
    A.ke = data.GlobalArray(ncomp=1)

    if not lattice:
        assert N % 2 == 0
        for px in range(N//2):
            pos = rng.uniform(low=-0.4999*E, high=0.4999*E, size=(1,3))
            cha = rng.uniform(low=-1., high=1.)

            A.P[px, :] = pos
            A.Q[px, 0] = cha

            A.P[-1*(px+1), :] = -1.0*pos
            A.Q[-1*(px+1), 0] = cha
    else:
        A.P[:] = utility.lattice.cubic_lattice((crn, crn, crn), (E,E,E))
        A.Q[:] = rng.uniform(low=-3.0, high=3.0, size=(N,1))

    bias = np.sum(A.Q[:])
    A.Q[:,0] -= bias/N

    A.M[:] = 1.
    A.V[:] = rng.normal(loc=0.0, scale=0.1, size=(N,3))

    dipole = np.zeros(3)
    for px in range(N):
        dipole[:] += A.P[px,:]*A.Q[px,0]

    bias = np.sum(A.Q[:])
    if MPIRANK == 0:
        print("DIPOLE:\t", dipole, "TOTAL CHARGE:\t", bias)

    A.scatter_data_from(0)


    # phi_py = fmm(A.P, A.Q, forces=A.F, async=ASYNC)



    potaa_rc = 3.
    potaa_rn = potaa_rc * 1.1
    delta = potaa_rn - potaa_rc

    potaa = utility.potential.VLennardJones(
        epsilon=1.0,
        sigma=1.0,
        rc=potaa_rc
    )



    constants = [
        kernel.Constant('dt', dt),
        kernel.Constant('dht', 0.5*dt),
    ]

    vv_kernel1 = kernel.Kernel('vv1', vv_kernel1_code, constants)
    vv_p1 = ParticleLoop(
        kernel=vv_kernel1,
        dat_dict={'P': A.P(access.W),
                  'V': A.V(access.W),
                  'F': A.F(access.R),
                  'M': A.M(access.R)}
    )

    vv_kernel2 = kernel.Kernel('vv2', vv_kernel2_code, constants)
    vv_p2 = ParticleLoop(
        kernel=vv_kernel2,
        dat_dict={'V': A.V(access.W),
                  'F': A.F(access.R),
                  'M': A.M(access.R),
                  'k': A.ke(access.INC0)}
    )

    potaa_force_updater = Pairloop(
        kernel=potaa.kernel,
        dat_dict=potaa.get_data_map(
            positions=A.P,
            forces=A.F,
            potential_energy=A.u
        ),
        shell_cutoff=potaa_rn
    )

    ke_list = []
    u_list = []
    q_list = []
    it_list = []


    start = time.time()
    for it in method.IntegratorRange(
            steps, dt, A.V, shell_steps, delta, verbose=False):

        # velocity verlet 1
        vv_p1.execute(A.npart_local)

        # vdw
        A.F[:,:] = 0
        potaa_force_updater.execute()
        qpot = fmm(positions=A.P, charges=A.Q, forces=A.F)



        # velocity verlet 2
        vv_p2.execute(A.npart_local)

        if it % 1 == 0:
            it_list.append(it)
            ke_list.append(A.ke[0])
            u_list.append(A.u[0])
            if MPIRANK == 0:
                print("{: 5d} {: 10.8e} {: 10.8e} {: 10.8e} {: 10.8e} | {: 8.4f} {: 8.4f}".format(
                    it, A.ke[0], A.u[0], qpot, A.ke[0] + A.u[0] + qpot,
                    fmm.flop_rate_mtl()/(10.**9), fmm.cuda_flop_rate_mtl()/(10.**9))
                )
    end = time.time()



    if MPIRANK == 0 and DEBUG:
        print(60*"-")
        print("Loop time:\t", end - start)
        print(60*"-")
        opt.print_profile()
        print(60*"-")


