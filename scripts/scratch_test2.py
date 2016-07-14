#!/usr/bin/python
import numpy as np
from ppmd import *


PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
State = state.State

N = 16
E = 8.
Eo2 = E/2.

A = State()
A.npart = N

A.domain = domain.BaseDomainHalo(extent=(E,E,E))

A.p = PositionDat(ncomp=3)
A.v = ParticleDat(ncomp=3)
A.f = ParticleDat(ncomp=3)
A.u = ScalarArray(ncomp=2)
A.u.halo_aware = True


A.p[:] = np.random.uniform(-1.*Eo2, Eo2, [N,3])
A.v[:] = np.random.normal(0, 2, [N,3])
A.f[:] = np.zeros([N,3])



A.scatter_data_from(0)


ljp = potential.LennardJones(sigma=1.0, epsilon=1.0, rc=2.5)
ljmap = ljp.get_data_map(positions=A.p, forces=A.f, potential_energy=A.u)


lj = pairloop.PairLoopNeighbourList(potential=ljp, 
                                    dat_dict=ljmap, 
                                    shell_cutoff=2.5)
lj.execute()


g_f = A.f.copy()
g_f.gather_data_on(0)


















