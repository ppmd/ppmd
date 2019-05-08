#!/usr/bin/python

import pytest

import numpy as np

import ppmd
from ppmd.access import *
from ppmd.data import ParticleDat, ScalarArray, GlobalArray, PositionDat, data_movement
from ppmd.kernel import Kernel

from ppmd.pairloop import CellByCellOMP, SubCellByCellOMP, PairLoopNeighbourListNSOMP
from ppmd.loop import ParticleLoop, ParticleLoopOMP

State = ppmd.state.State
BoundaryTypePeriodic = ppmd.domain.BoundaryTypePeriodic
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





@pytest.fixture(
    scope="module",
    params=(CellByCellOMP, SubCellByCellOMP, PairLoopNeighbourListNSOMP)
)
def PL(request):
    return request.param

def test_state_modifier_2(PL):

    # Checks pair loop functionality with add/remove particles

    rng = np.random.RandomState(96713)
    common_rng = np.random.RandomState(59171)

    E = 4.0
    N = 50
    Eo2 = E * 0.5

    N2 = 2*N
    cutoff = E/8
    
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    
    A.P = PositionDat()
    A.GID = ParticleDat(ncomp=1, dtype=INT64)
    A.NNL = ParticleDat(ncomp=1, dtype=INT64)


    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N2, 3))
    gi = np.reshape(np.arange(N2), (N2,1))
    nnp = np.zeros_like(gi)

    active_ids = set(gi[:N, 0])
    inactive_ids = set(gi[N:, 0])

    with A.modify() as m:
        if MPIRANK == 0:
            m.add(
                {
                    A.P: pi[:N, :],
                    A.GID: gi[:N, :]
                }
            )


    def _direct():

        for ix in active_ids:
            tnn = 0
            ir = pi[ix, :]
            for jx in active_ids:
                if ix == jx: continue
                jr = pi[jx, :]
                r = np.zeros_like(ir)

                for dx in (0,1,2):
                    r[dx] = ir[dx] - jr[dx]
                    if r[dx] >=  Eo2: r[dx] -= E 
                    elif r[dx] <= -Eo2: r[dx] += E 

                if np.linalg.norm(r) <= cutoff:
                    tnn += 1

            nnp[ix, 0] = tnn
    
    
    nnl_kernel_src = """
    const double r0 = P.i[0] - P.j[0];
    const double r1 = P.i[1] - P.j[1];
    const double r2 = P.i[2] - P.j[2];
    const double rr = r0 * r0 + r1*r1 + r2*r2;
    if (rr <= {cutoff}) {{
        NNL.i[0]++;
    }}
    """.format(
        cutoff=cutoff * cutoff
    )

    nnl_kernel = Kernel('nnl_kernel', nnl_kernel_src)
    nnl = PL(
        kernel=nnl_kernel,
        dat_dict={
            'P': A.P(READ),
            'NNL': A.NNL(INC_ZERO)
        },
        shell_cutoff = 1.02 * cutoff
    )

    
    def _check():

        assert A.npart == len(active_ids)
        assert len(active_ids) + len(inactive_ids) == N2

        nnl.execute()
        _direct()

        for ix in range(A.npart_local):
            gid = A.GID[ix, 0]
            correct = nnp[gid, 0]
            lid = np.where(A.GID.view[:, 0] == gid)[0]
            to_test = A.NNL[lid, 0]
            assert correct == to_test
    

    _check()


    for testx in range(200):

        add_bool = common_rng.randint(0, 2)
        add_bool = False if (len(inactive_ids) == 0) else add_bool
        remove_bool = common_rng.randint(0, 2) and bool(A.npart)
        
        if remove_bool:

            gid_to_remove = sorted(active_ids)[common_rng.randint(0, len(active_ids))]
            active_ids.remove(gid_to_remove)
            inactive_ids.add(gid_to_remove)

            lid = np.where(A.GID.view[:, 0] == gid_to_remove)
            lid = lid[0][0] if len(lid[0]) > 0 else None

            with A.modify() as m:
                if lid is not None:
                    m.remove((lid,))

            _check()
        
        if add_bool:

            gid_to_add = sorted(inactive_ids)[common_rng.randint(0, len(inactive_ids))]
            active_ids.add(gid_to_add)
            inactive_ids.remove(gid_to_add) 

            ##pick a random rank to add the particle
            rank = common_rng.randint(0, MPISIZE)

            with A.modify() as m:
                if MPIRANK == rank:
                    m.add({
                        A.P: pi[gid_to_add, :],
                        A.GID: gi[gid_to_add, :],
                    })

            _check()


@pytest.fixture(
    scope="module",
    params=(ParticleLoop, ParticleLoopOMP)
)
def PL2(request):
    return request.param

def test_state_modifier_3(PL2):

    # Checks particle loop functionality with add/remove particles

    rng = np.random.RandomState(96713)
    common_rng = np.random.RandomState(59171)

    E = 4.0
    N = 50
    Eo2 = E * 0.5

    N2 = 2*N
    cutoff = E/8
    
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    
    A.P = PositionDat()
    A.GID = ParticleDat(ncomp=1, dtype=INT64)
    A.NNL = ParticleDat(ncomp=1, dtype=INT64)

    A.VALUE = ScalarArray(ncomp=1, dtype=INT64)
    A.VALUE[0] = 42


    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N2, 3))
    gi = np.reshape(np.arange(N2), (N2,1))

    active_ids = set(gi[:N, 0])
    inactive_ids = set(gi[N:, 0])

    with A.modify() as m:
        if MPIRANK == 0:
            m.add(
                {
                    A.P: pi[:N, :],
                    A.GID: gi[:N, :]
                }
            )

    nnl_kernel_src = """
    NNL.i[0] = VALUE[0] * GID.i[0];
    """
    nnl_kernel = Kernel('nnl2_kernel', nnl_kernel_src)
    nnl = PL2(
        kernel=nnl_kernel,
        dat_dict={
            'GID': A.GID(READ),
            'VALUE': A.VALUE(READ),
            'NNL': A.NNL(WRITE)
        }
    )

    
    def _check():

        assert A.npart == len(active_ids)
        assert len(active_ids) + len(inactive_ids) == N2

        nnl.execute()

        for ix in range(A.npart_local):
            assert A.NNL.view[ix, 0] == A.GID.view[ix, 0] * A.VALUE[0]

    _check()


    for testx in range(200):

        A.VALUE[0] = common_rng.randint(0, 100000)

        add_bool = common_rng.randint(0, 2)
        add_bool = False if (len(inactive_ids) == 0) else add_bool
        remove_bool = common_rng.randint(0, 2) and bool(A.npart)
        
        if remove_bool:

            gid_to_remove = sorted(active_ids)[common_rng.randint(0, len(active_ids))]
            active_ids.remove(gid_to_remove)
            inactive_ids.add(gid_to_remove)

            lid = np.where(A.GID.view[:, 0] == gid_to_remove)
            lid = lid[0][0] if len(lid[0]) > 0 else None

            with A.modify() as m:
                if lid is not None:
                    m.remove((lid,))

            _check()
        
        if add_bool:

            gid_to_add = sorted(inactive_ids)[common_rng.randint(0, len(inactive_ids))]
            active_ids.add(gid_to_add)
            inactive_ids.remove(gid_to_add) 

            ##pick a random rank to add the particle
            rank = common_rng.randint(0, MPISIZE)

            with A.modify() as m:
                if MPIRANK == rank:
                    m.add({
                        A.P: pi[gid_to_add, :],
                        A.GID: gi[gid_to_add, :],
                    })

            _check()













