#!/usr/bin/python

import ctypes
import numpy as np


import ppmd as md

N = 1000

rank = md.mpi.MPI_HANDLE.rank


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray



def test_host_all_to_all_NS():
    if rank == 4:
        A = ParticleDat(
            npart=1000,
            ncomp=1,
            dtype=ctypes.c_int64
        )
        B = ParticleDat(
            npart=1000,
            ncomp=1,
            dtype=ctypes.c_int64
        )


        A[:,0] = np.arange(N)
        B[:] = 0

        k = md.kernel.Kernel(
            'AllToAll_1',
            '''
            B.i[0] += A.j[0];
            '''
        )

        p1 = md.pairloop.AllToAllNS(
            kernel=k,
            dat_dict={
                'A': A(md.access.R),
                'B': B(md.access.W)
                }
        )

        p1.execute()


        # check output
        sum = np.sum(np.arange(N))
        C = sum - np.arange(N)

        for i in range(N):
            assert B[i] == C[i]





def test_host_all_to_all():
    if rank == 0:
        A = ParticleDat(
            npart=1000,
            ncomp=1,
            dtype=ctypes.c_int64
        )
        B = ParticleDat(
            npart=1000,
            ncomp=1,
            dtype=ctypes.c_int64
        )

        A[:,0] = np.arange(N)
        B[:] = 0

        k = md.kernel.Kernel(
            'AllToAll_1',
            '''
            B.i[0] += A.j[0];
            B.j[0] += A.i[0];
            '''
        )


        p1 = md.pairloop.AllToAll(
            kernel=k,
            dat_dict={
                'A': A(md.access.R),
                'B': B(md.access.W)
                }
        )

        p1.execute()


        # check output
        sum = np.sum(np.arange(N))
        C = sum - np.arange(N)

        for i in range(N):
            assert B[i] == C[i]
