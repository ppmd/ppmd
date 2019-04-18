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

MPI = ppmd.mpi.MPI
MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.size
MPIBARRIER = MPI.COMM_WORLD.Barrier

import sys

def test_state_modifier_1():
    
    rng = np.random.RandomState(97531)

    E = 4.0
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    

    A.P = PositionDat()
    A.V = ParticleDat(ncomp=3)

    

    with A.modify() as m:

        m.add(
            {
                A.P: np.array((1, 1, 0)),
                A.V: np.array((0.1, 0.3, 0.1))
            }
        )
 

    with A.modify() as m:
        if A.npart_local == 2:
            m.remove((0,1))


    if MPIRANK == 0:
        print('\n')
    MPIBARRIER()

    for rk in range(MPISIZE):
        if MPIRANK == rk:
            print(A.domain.comm.rank, A.npart, A.npart_local)
            print(A.P.view)
            print(A.V.view)
            sys.stdout.flush()
        MPIBARRIER()
        if MPIRANK == 0:
            print("-" * 80)
            sys.stdout.flush()       
        MPIBARRIER()

    MPIBARRIER()






