import pytest

from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

import ctypes

REAL = ctypes.c_double
INT64 = ctypes.c_int64

import numpy as np


from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()

from ppmd.coulomb.direct import *


def test_free_space_1():

    FSD = FreeSpaceDirect()

    for testx in range(500):

        rng = np.random.RandomState(seed=(MPIRANK+1)*93573)
        N = rng.randint(1, 100)
        
        ppi = np.zeros((N, 3), REAL)
        qi = np.zeros((N, 1), REAL)

        def _direct():
            _phi_direct = 0.0
            # compute phi from image and surrounding 26 cells
            for ix in range(N):
                for jx in range(ix+1, N):
                    rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                    _phi_direct += qi[ix, 0] * qi[jx, 0] / rij
            return _phi_direct


        ppi[:] = rng.uniform(-1.0, 1.0, (N,3))
        qi[:] = rng.uniform(-1.0, 1.0, (N,1))

        phi_py = _direct()
        phi_c = FSD(N, ppi, qi)

        rel = abs(phi_py)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_py - phi_c) / rel
        assert err < 10.**-14




def test_nearest_1():

    tuples = tuple(product(range(-1, 2), range(-1, 2), range(-2, 3)))

    E = np.array((39., 71., 51.))
    ND = NearestDirect(E, tuples)


    for testx in range(max(10, 20//MPISIZE)):

        rng = np.random.RandomState(seed=(MPIRANK+1)*93573)
        N = rng.randint(1, 100)


        ppi = np.zeros((N, 3), REAL)
        qi = np.zeros((N, 1), REAL)

        def _direct():
            _phi_direct = 0.0
            # compute phi from image and surrounding 26 cells
            for ix in range(N):
                for jx in range(ix+1, N):
                    rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                    _phi_direct += qi[ix, 0] * qi[jx, 0] / rij

                for jx in range(N):
                    for ox in tuples:
                        if ox[0] != 0 or ox[1] != 0 or ox[2] != 0:
                            rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:] + (np.multiply(E, np.array(ox))))
                            _phi_direct += 0.5 * qi[ix, 0] * qi[jx, 0] / rij

            return _phi_direct


        ppi[:] = rng.uniform(-1.0, 1.0, (N,3))
        qi[:] = rng.uniform(-1.0, 1.0, (N,1))

        phi_py = _direct()
        phi_c = ND(N, ppi, qi)

        rel = abs(phi_py)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_py - phi_c) / rel

        assert err < 10.**-13


def test_pbc_1():

    E = 19.
    L = 16
    N = 10
    rc = E/4

    rng  = np.random.RandomState(seed=8123)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.F = data.ParticleDat(ncomp=3)
    A.G = data.ParticleDat(ncomp=1, dtype=INT64)

    pi = np.zeros((N, 3), REAL)
    qi = np.zeros((N, 1), REAL)
    gi = np.zeros((N, 1), INT64)
    
    gi[:, 0] = np.arange(N)
    pi[:] = rng.uniform(-0.5*E, 0.5*E, (N, 3))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi)/N
    qi[:] -= bias

    A.P[:] = pi
    A.Q[:] = qi
    A.G[:] = gi

    A.scatter_data_from(0)

    EWALD = EwaldOrthoganalHalf(domain=A.domain, real_cutoff=rc, shared_memory='omp', eps=10.**-8)
    PBCD = PBCDirect(E, A.domain, L)

    def _check1():

        return EWALD(positions=A.P, charges=A.Q, forces=A.F)
    
    phi_f = _check1()

    phi_c = PBCD(N, pi, qi)
    rel = abs(phi_f)
    rel = 1.0 if rel == 0 else rel
    err = abs(phi_c - phi_f) / rel    
    assert err < 10.**-5
 
    for testx in range(100):

        pi[:] = rng.uniform(-0.5*E, 0.5*E, (N, 3))
        with A.P.modify_view() as m:
            for px in range(A.npart_local):
                g = A.G[px, 0]
                A.P[px, :] = pi[g, :]
    
        phi_f = _check1()

        phi_c = PBCD(N, pi, qi)
        rel = abs(phi_f)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_c - phi_f) / rel    
        assert err < 10.**-4
         


def test_pbc_2():

    E = (50., 40, 30)

    L = 12
    N = 40
    rc = np.min(E)/4

    rng  = np.random.RandomState(seed=8123)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=E)
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.F = data.ParticleDat(ncomp=3)
    A.G = data.ParticleDat(ncomp=1, dtype=INT64)

    pi = np.zeros((N, 3), REAL)
    qi = np.zeros((N, 1), REAL)
    gi = np.zeros((N, 1), INT64)
    
    gi[:, 0] = np.arange(N)
    for dx in (0,1,2):
        pi[:, dx] = rng.uniform(-0.5*E[dx], 0.5*E[dx], N)

    
    if N == 8:
        pi[:,:] = (
            (-1.0, -1.0, -1.0),
            ( 1.0, -1.0, -1.0),
            ( 1.0,  1.0, -1.0),
            (-1.0,  1.0, -1.0),
            (-1.0, -1.0,  1.0),
            ( 1.0, -1.0,  1.0),
            ( 1.0,  1.0,  1.0),
            (-1.0,  1.0,  1.0),            
        )


    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi)/N
    qi[:] -= bias

    if N == 8:
        qi[:,0] = (
            ( 1.0),
            (-1.0),
            ( 1.0),
            (-1.0),
            (-1.0),
            ( 1.0),
            (-1.0),
            ( 1.0),            
        )


    A.P[:] = pi
    A.Q[:] = qi
    A.G[:] = gi

    A.scatter_data_from(0)

    EWALD = EwaldOrthoganalHalf(domain=A.domain, real_cutoff=rc, shared_memory='omp', eps=10.**-10)
    PBCD = PBCDirect(E, A.domain, L)

    def _check1():

        return EWALD(positions=A.P, charges=A.Q, forces=A.F)
    
    phi_f = _check1()

    phi_c = PBCD(N, pi, qi)
    rel = abs(phi_f)
    rel = 1.0 if rel == 0 else rel
    err = abs(phi_c - phi_f) / rel    
    assert err < 10.**-5
 
    for testx in range(100):

        for dx in (0,1,2):
            pi[:, dx] = rng.uniform(-0.5*E[dx], 0.5*E[dx], N)

        with A.P.modify_view() as m:
            for px in range(A.npart_local):
                g = A.G[px, 0]
                A.P[px, :] = pi[g, :]
    
        phi_f = _check1()

        phi_c = PBCD(N, pi, qi)
        rel = abs(phi_f)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_c - phi_f) / rel  
        assert err < 10.**-4



