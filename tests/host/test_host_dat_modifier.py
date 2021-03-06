#!/usr/bin/python

import numpy as np

import ppmd
from ppmd.access import *
from ppmd.data import ParticleDat, ScalarArray, GlobalArray, PositionDat, data_movement

State = ppmd.state.State
BoundaryTypePeriodic = ppmd.domain.BoundaryTypePeriodic
BaseDomainHalo = ppmd.domain.BaseDomainHalo

import ctypes
INT64 = ctypes.c_int64


def test_non_pos_1():
    
    rng = np.random.RandomState(97531)

    E = 1.0
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    

    N = 10
    A.npart = 10

    A.P = PositionDat()
    A.V = ParticleDat(ncomp=3)
    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.V[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.scatter_data_from(0)

    A.post = ParticleDat(ncomp=1, dtype=INT64)
    assert A.post.npart_local == A.P.npart_local

    
    curr_int_id = A.V._vid_int
    new_vel = rng.uniform(size=(A.npart_local, 3))

    with A.V.modify_view() as m:
        assert m.shape[0] == A.npart_local
        assert m.shape[1] == 3
        m[:] = new_vel

    new_int_id = A.V._vid_int
    assert new_int_id > curr_int_id

    if A.npart_local > 0:
        assert np.linalg.norm(A.V[:A.npart_local,:] - new_vel, np.inf) < 10.**-16


    with A.V.modify_view() as m:
        pass
    assert A.V._vid_int > new_int_id
    


def test_pos_1():
    
    rng = np.random.RandomState(97531)

    E = 1.0
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    
    N = 1
    A.npart = N

    M = 4000

    A.P = PositionDat()
    A.GID = ParticleDat(ncomp=1, dtype=INT64)
    A.V = ParticleDat(ncomp=3)
    
    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    vi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    gi = rng.randint(0, 1000, size=(N,1))
    A.P[:] = pi
    A.V[:] = vi
    A.GID[:] = gi

    A.scatter_data_from(0)
    
    assert N == 1


    for testx in range(M):
        
        pnew = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))

        on_edge = rng.binomial(1, 0.1)
        if on_edge:

            edge_coeffs = (
                (-0.5*E, 0.5*E, pnew[0, 0]),
                (-0.5*E, 0.5*E, pnew[0, 1]),
                (-0.5*E, 0.5*E, pnew[0, 2]),
            )

            coeff_inds = rng.randint(0, 2, 3)
            pnew = np.array((
                edge_coeffs[0][coeff_inds[0]],
                edge_coeffs[1][coeff_inds[1]],
                edge_coeffs[2][coeff_inds[2]]
                ))



        int_id = A.P._vid_int
        with A.P.modify_view() as m:
            m[:] = pnew
        assert int_id < A.P._vid_int

        if A.npart_local == 1:
            assert A.P[0, 0] >= A.domain.boundary[0]
            assert A.P[0, 0] <= A.domain.boundary[1]
            assert A.P[0, 1] >= A.domain.boundary[2]
            assert A.P[0, 1] <= A.domain.boundary[3]
            assert A.P[0, 2] >= A.domain.boundary[4]
            assert A.P[0, 2] <= A.domain.boundary[5]

            assert np.linalg.norm(A.P[0,:] - pnew, np.inf) < 10.**-16
            assert np.linalg.norm(A.V[0,:] - vi, np.inf) < 10.**-16
            assert A.GID[0,0] == gi[0, 0]
        
        else:

            assert A.npart_local == 0

        lcount = np.array((A.npart_local,), np.int)
        gcount = np.zeros_like(lcount)
        A.domain.comm.Allreduce(lcount, gcount)
        assert gcount[0] == N







def test_pos_2():
    
    rng = np.random.RandomState(97531)

    E = 1.0
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    
    N = 1000
    A.npart = N

    M = 200

    A.P = PositionDat()
    A.GID = ParticleDat(ncomp=1, dtype=INT64)
    A.V = ParticleDat(ncomp=3)
    
    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    vi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    gi = np.arange(N)
    A.P[:] = pi
    A.V[:] = vi
    A.GID[:, 0] = gi

    A.scatter_data_from(0)
    

    for testx in range(M):
        
        pnew = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))

        with A.P.modify_view() as m:
            m[:] = pnew[A.GID.view[:, 0], :]

        lcount = np.array((A.npart_local,), np.int)
        gcount = np.zeros_like(lcount)
        A.domain.comm.Allreduce(lcount, gcount)
        assert gcount[0] == N

        for px in range(A.npart_local):
            assert np.linalg.norm(A.P[px, :] - pnew[A.GID[px, 0], :], np.inf) < 10.**-16
            assert np.linalg.norm(A.V[px, :] - vi[A.GID[px, 0], :], np.inf) < 10.**-16





def test_pos_3():
    """
    Test a large number of particles being sent to one rank
    (Had issues with Get_accumalate seg faulting on fence with
    OpenMPI).
    """
    
    rng = np.random.RandomState(97531)

    E = 1.0
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    
    N = 100000
    A.npart = N

    A.P = PositionDat()
    A.GID = ParticleDat(ncomp=1, dtype=INT64)
    A.V = ParticleDat(ncomp=3)
    
    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    vi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    gi = np.arange(N)
    A.P[:] = pi
    A.V[:] = vi
    A.GID[:, 0] = gi

    A.scatter_data_from(0)
    
    pnew = np.zeros((N,3))
    pnew[:,:] = 0.1


    with A.P.modify_view() as m:
        m[:] = pnew[A.GID.view[:, 0], :]

    lcount = np.array((A.npart_local,), np.int)
    gcount = np.zeros_like(lcount)
    A.domain.comm.Allreduce(lcount, gcount)
    assert gcount[0] == N

    for px in range(A.npart_local):
        assert np.linalg.norm(A.P[px, :] - pnew[A.GID[px, 0], :], np.inf) < 10.**-16
        assert np.linalg.norm(A.V[px, :] - vi[A.GID[px, 0], :], np.inf) < 10.**-16



