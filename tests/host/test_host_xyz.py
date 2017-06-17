#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md
from mpi4py import MPI

N = 16
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()

PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
State = md.state.State

@pytest.fixture
def state():
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param


def test_host_xyz_rw(tmpdir):
    N2 = 1000
    a = np.random.uniform(size=[N2,3])*100. - 50.
    filename  = str(tmpdir) + '/test_xyz_'+str(rank)+'.xyz'
    md.utility.xyz.numpy_to_xyz(a, filename)

    xyz_reader = md.utility.xyz.XYZ(filename)
    b = xyz_reader.positions
    a = a.ravel()
    b = b.ravel()
    for ix in range(3*N2):
        assert abs(b[ix] - a[ix]) < 10.**-15
    




    
