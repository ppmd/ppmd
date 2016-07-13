#!/usr/bin/python
import numpy as np
import sys
import ctypes

from ppmd import *
from ppmd.cuda import *

PositionDat = cuda_data.PositionDat
ParticleDat = cuda_data.ParticleDat
ScalarArray = cuda_data.ScalarArray
State = cuda_state.State
N = 1000000
E = 8.
Eo2 = E/2.


A = State()
A.npart = N


A.domain = domain.BaseDomainHalo(extent=(E,E,E))


A.p = PositionDat(ncomp=3)
A.v = ParticleDat(ncomp=3)
A.f = ParticleDat(ncomp=3)
A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

A.u = ScalarArray(ncomp=2)


A.u.halo_aware = True


A.p[:] = np.random.uniform(-1.*Eo2, Eo2, [N,3])
A.v[:] = np.random.normal(0, 2, [N,3])
A.f[:] = np.zeros([N,3])
A.gid[:,0] = np.arange(A.npart)




for rk in range(mpi.MPI_HANDLE.nproc):
    if mpi.MPI_HANDLE.rank == rk:
        #print rk, A.gid[:]
        sys.stdout.flush()
    mpi.MPI_HANDLE.comm.barrier()

A.scatter_data_from(0)


for rk in range(mpi.MPI_HANDLE.nproc):
    if mpi.MPI_HANDLE.rank == rk:
        #print rk, A.gid[:A.npart_local:]
        sys.stdout.flush()
    mpi.MPI_HANDLE.comm.barrier()

A.gather_data_on(0)


if mpi.MPI_HANDLE.rank == 0:
    ret = A.gid[:,0]
    ret.sort()

    ret_true = np.arange(A.npart)

    print " Test passed ==", A.npart == np.sum(ret==ret_true)








# 
# 
# ljp = potential.LennardJones(sigma=1.0, epsilon=1.0, rc=2.5)
# ljmap = ljp.get_data_map(positions=A.p, forces=A.f, potential_energy=A.u)
# 
# 
# lj = pairloop.PairLoopNeighbourList(potential=ljp, 
#                                     dat_dict=ljmap, 
#                                     shell_cutoff=2.5)
# lj.execute()
# 
# 
# g_f = A.f.copy()
# g_f.gather_data_on(0)


















