#!/usr/bin/python
import numpy as np

from ppmd import *

N = 10

A = state.State()
A.npart = N

A.domain = domain.BaseDomainHalo(extent=(12.,12.,12.))

A.p = data.PositionDat(ncomp=3)
A.v = data.ParticleDat(ncomp=3)


A.p[::] = np.random.uniform(-6, 6, [N,3])
A.v[::] = np.random.normal(0, 2, [N,3])
A.broadcast_data_from(0)



A.filter_on_domain_boundary()


#plot_dat = A.p.snapshot()
#plot_dat.gather_data_on(0)
















