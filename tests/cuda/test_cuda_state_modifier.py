#!/usr/bin/python

import numpy as np

import ppmd
from ppmd.access import *
from ppmd.cuda.cuda_data import ParticleDat, ScalarArray, GlobalArray, PositionDat


State = ppmd.cuda.cuda_state.State

BoundaryTypePeriodic = ppmd.cuda.cuda_domain.BoundaryTypePeriodic
BaseDomainHalo = ppmd.domain.BaseDomainHalo

import ctypes
INT64 = ctypes.c_int64

MPI = ppmd.mpi.MPI
MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.size
MPIBARRIER = MPI.COMM_WORLD.Barrier

import sys

def test_state_modifier_1():
    
    rng = np.random.RandomState(97531)
    common_rng = np.random.RandomState(5932)

    E = 4.0
    N = 20
    
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    
    A.P = PositionDat()
    A.GID = ParticleDat(ncomp=1, dtype=INT64)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    gi = np.reshape(np.arange(N), (N,1))
    
    # test adding particle data
    with A.modify() as m:
        if MPIRANK == 0:
            m.add(
                {
                    A.P: pi,
                    A.GID: gi
                }
            )

    for lx in range(A.npart_local):
        # implicitly checks global id in GID
        gid = A.GID[lx, 0]
        assert np.linalg.norm(A.P[lx, :] - pi[gid, :], np.inf) < 10.**-16

    # global count
    assert A.npart == N

    # remove some particles based on GID
    gids = common_rng.permutation(range(N))

    for rx in range(N):
        gid = gids[rx]
        old_npart = A.npart
        with A.modify() as m:
            lid = np.where(A.GID.view[:, 0] == gid)
            assert len(lid) < 2
            if len(lid) == 1:
                lid = lid[0]
                m.remove(lid)
        assert A.npart == old_npart - 1

        # check the data for the remaining particles is correct
        for lx in range(A.npart_local):
            # implicitly checks global id in GID
            gid2 = A.GID[lx, 0]
            # this should have been removed
            assert gid != gid2
            assert np.linalg.norm(A.P[lx, :] - pi[gid2, :], np.inf) < 10.**-16

    assert A.npart == 0
    
    pi2 = rng.uniform(low=-0.5*E, high=0.5*E, size=(N*2, 3))
    gi2 = np.reshape(np.arange(N*2), (N*2,1))
    
    # re-add N in one go
    with A.modify() as m:
        if MPIRANK == 0:
            m.add(
                {
                    A.P: pi2[:N, :],
                    A.GID: gi2[:N, :]
                }
            )

    assert A.npart == N

    # remove and add particles
    for rx in range(N):
        gid = gids[rx]
        with A.modify() as m:
            # remove a particle
            lid = np.where(A.GID.view[:, 0] == gid)
            assert len(lid) < 2
            if len(lid) == 1:
                lid = lid[0]
                m.remove(lid)

            # add a particle
            if MPIRANK == (rx % MPISIZE):
                m.add(
                    {
                        A.P: pi2[N + rx, :],
                        A.GID: gi2[N + rx, :]
                    }
                )

        assert A.npart == N

        # check the data for the remaining particles is correct
        for lx in range(A.npart_local):
            # implicitly checks global id in GID
            gid2 = A.GID[lx, 0]
            # this should have been removed
            assert np.linalg.norm(A.P[lx, :] - pi2[gid2, :], np.inf) < 10.**-16









