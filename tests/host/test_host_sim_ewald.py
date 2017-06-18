#!/usr/bin/python

import pytest
import ctypes
import numpy as np
from math import sqrt, erfc, exp, pi
import os



import ppmd as md

VERBOSE = True
SHARED_MEMORY = False


rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
State = md.state.State
ParticleLoop = md.loop.ParticleLoop
Pairloop = md.pairloop.PairLoopNeighbourListNS
PBC = md.domain.BoundaryTypePeriodic()
GlobalArray = md.data.GlobalArray
State = md.state.State


vv_kernel1_code = '''
const double M_tmp = 1.0/M.i[0];
V.i[0] += dht*F.i[0]*M_tmp;
V.i[1] += dht*F.i[1]*M_tmp;
V.i[2] += dht*F.i[2]*M_tmp;
P.i[0] += dt*V.i[0];
P.i[1] += dt*V.i[1];
P.i[2] += dt*V.i[2];
'''

vv_kernel2_code = '''
const double M_tmp = 1.0/M.i[0];
V.i[0] += dht*F.i[0]*M_tmp;
V.i[1] += dht*F.i[1]*M_tmp;
V.i[2] += dht*F.i[2]*M_tmp;
k[0] += (V.i[0]*V.i[0] + V.i[1]*V.i[1] + V.i[2]*V.i[2])*0.5*M.i[0];
'''


@pytest.fixture(
    scope="module",
    params=(
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)
    )
)
def directiong(request):
    return request.param

@pytest.fixture(
    scope="module",
    params=(
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),

        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),

        (0, 1, 1),
        (0, 1, -1),
        (0, -1, 1),
        (0, -1, -1),

        (1, 0, 1),
        (1, 0, -1),
        (-1, 0, 1),
        (-1, 0, -1),

        (1, 1, 1),
        (-1, 1, 1),
        (-1, -1, 1),
        (1, -1, 1),

        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, -1)

    )
)
def directiona(request):
    return request.param

#@pytest.mark.skip
def test_host_sim_1(directiong):

    A = State()
    dt = 0.001
    steps = 2250
    shell_steps = 10
    v = 4.0
    A.npart = 2
    E = 80.
    extent = (E, E, E)
    A.domain = md.domain.BaseDomainHalo(extent=extent)
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    # init state
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.f2 = ParticleDat(ncomp=3)
    A.mass = ParticleDat(ncomp=1)
    A.q = ParticleDat(ncomp=1)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=1)
    A.ke = ScalarArray(ncomp=1)
    A.rr = ScalarArray(ncomp=1)
    A.rl = ScalarArray(ncomp=1)

    A.gid[:, 0] = np.arange(A.npart)
    A.p[0, :] = 1.*np.array(directiong)
    A.p[1, :] = -1.*np.array(directiong)
    A.v[:, 0] = directiong[0] * v
    A.v[:, 1] = directiong[1] * v
    A.v[:, 2] = directiong[2] * v
    A.f[:] = 0.0
    A.u[0] = 0.0
    A.ke[0] = 0.0
    A.q[:, 0] = 1.0

    A.mass[:, 0] = 40.

    A.npart_local = A.npart
    A.filter_on_domain_boundary()


    c = md.coulomb.ewald.EwaldOrthoganal(
        domain=A.domain,
        real_cutoff=12.,
        shared_memory=SHARED_MEMORY
    )

    A.f2[:] = 0.0
    c.extract_forces_energy_real(A.p, A.q, A.f, A.rr)
    c.evaluate_contributions(A.p, A.q)
    c.extract_forces_energy_reciprocal(A.p, A.q, A.f2, A.rl)

    rij = np.linalg.norm(A.p[0, :] - A.p[1, :])
    assert abs(
        A.rr[0] - \
        0.5**A.q[0]*A.q[1]*\
        erfc(sqrt(c.alpha)*rij)
    ) < 10.**-12

    A.gather_data_on(0)

    irij = 1.0/rij

    if rank == 0:
        factor = -1.0*A.q[0]*A.q[1]*irij*(
            (sqrt(c.alpha/pi)*-2.0)*exp(-1.0*c.alpha*(rij**2.)) -
            irij*erfc(sqrt(c.alpha)*rij)
        )

        inds = A.gid[:, 0].argsort()
        fs = A.f[inds]
        f2s = A.f2[inds]

        assert np.sum(abs(fs[0, :] - factor*directiong)) < 10.**-12
        assert np.sum(abs(fs[1, :] + factor*directiong)) < 10.**-12

        not_dir = np.logical_not(np.array(np.abs(directiong), dtype=bool))
        dir = np.array(np.abs(directiong), dtype=bool)

        # checks symmetric long range parts are zero
        assert np.sum(np.abs(f2s[:, not_dir])) < 10.**-16
        assert np.sum(np.abs(f2s[:, not_dir])) < 10.**-16

        # checks sign of long range part
        assert np.sign(f2s[0, dir]) == np.sign(np.array(directiong)[dir])
        assert np.sign(f2s[1, dir]) == -1*np.sign(np.array(directiong)[dir])



















