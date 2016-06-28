#!/usr/bin/python
import numpy as np
import sys


from ppmd import *


N = 16
E = 12.
Eo2 = E/2.

A = state.State()
A.npart = N

A.domain = domain.BaseDomainHalo(extent=(E,E,E))

A.p = data.PositionDat(ncomp=3)
A.v = data.ParticleDat(ncomp=3)
A.f = data.ParticleDat(ncomp=3)
A.u = data.ScalarArray(ncomp=2)
A.u.halo_aware = True


A.p[::] = np.random.uniform(-1.*Eo2, Eo2, [N,3])
A.v[::] = np.random.normal(0, 2, [N,3])
A.f[::] = np.zeros([N,3])
A.broadcast_data_from(0)
A.filter_on_domain_boundary()



ljp = potential.LennardJones(sigma=1.0, epsilon=1.0, rc=2.5)
ljmap = ljp.get_data_map(positions=A.p, forces=A.f, potential_energy=A.u)



lj = pairloop.PairLoopNeighbourList(potential=ljp, 
                                    dat_dict=ljmap, 
                                    shell_cutoff=2.75)
lj.execute()


g_f = A.f.snapshot()


g_f.gather_data_on(0)


for rx in range(mpi.MPI_HANDLE.nproc):
    if mpi.MPI_HANDLE.rank == rx:
        print mpi.MPI_HANDLE.rank
        print A.f[:A.f.npart:]
        print 80*"-"
        sys.stdout.flush()
    mpi.MPI_HANDLE.barrier()


if mpi.MPI_HANDLE.rank == 0:
    print 80*"="
    #print A.f
    print g_f
    

















