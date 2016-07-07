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
print "Init"
N = 16
E = 8.
Eo2 = E/2.


A = State()
A.npart = N

print "state made"

A.domain = domain.BaseDomainHalo(extent=(E,E,E))

print "domain added"

A.p = PositionDat(ncomp=3)
A.v = ParticleDat(ncomp=3)
A.f = ParticleDat(ncomp=3)
A.u = ScalarArray(ncomp=2)

print "dats added"

A.u.halo_aware = True


A.p[:] = np.random.uniform(-1.*Eo2, Eo2, [N,3])
A.v[:] = np.random.normal(0, 2, [N,3])
A.f[:] = np.zeros([N,3])


N2 = 10000
tmp = cuda_data.ScalarArray(ncomp=N2, dtype=ctypes.c_int)
tmp[:] = (1+mpi.MPI_HANDLE.rank)*np.array(range(N2))

npscan = np.cumsum(tmp[:-1:])[-1]
cuda_runtime.LIB_CUDA_MISC['cudaExclusiveScanInt'](tmp.ctypes_data, ctypes.c_int(N2))
print tmp[-1] == npscan, tmp[-1], npscan


quit()



for rk in range(mpi.MPI_HANDLE.nproc):
    if mpi.MPI_HANDLE.rank == rk:
        print rk, A.v[:]
        sys.stdout.flush()
    mpi.MPI_HANDLE.comm.barrier()

# A.scatter_data_from(0)
A.broadcast_data_from(0)






#A.p.broadcast_data_from(0)
#cuda_mpi.cuda_mpi_err_check(0)

for rk in range(mpi.MPI_HANDLE.nproc):
    if mpi.MPI_HANDLE.rank == rk:
        print rk, A.v[:]
        sys.stdout.flush()
    mpi.MPI_HANDLE.comm.barrier()





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


















