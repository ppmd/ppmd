__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import ctypes
import numpy as np

import pytest

import ppmd as md

mpi_rank = md.mpi.MPI.COMM_WORLD.Get_rank()
mpi_size = md.mpi.MPI.COMM_WORLD.Get_size()
ParticleDat = md.data.ParticleDat
PositionDat = md.data.PositionDat
ScalarArray = md.data.ScalarArray
State = md.state.BaseMDState
GlobalArray = md.data.GlobalArray

def assert_tol(val, tol, msg="tolerance not met"):
    assert abs(val) < 10.**(-1*tol), msg

def test_ewald_energy_python_co2_2_1():
    """
    Test that the python implementation of ewald calculates the correct 
    real space contribution and self interaction contribution.
    """

    eta = 0.26506
    alpha = eta**2.
    rc = 12.

    e = 24.47507
    meo2 = -0.5 * e

    data = np.load('../res/coulomb/CO2.npy')

    N = data.shape[0]
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    c = md.coulomb.ewald.EwaldOrthoganal(domain=A.domain, real_cutoff=rc, alpha=alpha)
    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"


    A.positions = PositionDat(ncomp=3)
    A.forces = ParticleDat(ncomp=3)
    A.charges = ParticleDat(ncomp=1)

    energy = GlobalArray(size=1, dtype=ctypes.c_double)

    if mpi_rank == 0:
        A.positions[:] = data[:,0:3:]
        A.charges[:, 0] = data[:,3]
    A.scatter_data_from(0)

    c.evaluate_contributions(positions=A.positions, charges=A.charges)

    energy[0] = 0.0
    c.extract_forces_energy(A.positions, A.charges, A.forces, energy)

    assert abs(energy[0]*c.internal_to_ev() - 0.917463161E1) < 10.**-3

    A.gather_data_on(0)
    if mpi_rank == 0:
        assert abs( np.sum(A.charges[:, 0]) ) < 10.**-12, "total charge not zero"


def test_ewald_energy_python_co2_2_2():
    eta = 0.26506
    alpha = eta**2.
    rc = 12.

    e = 24.47507

    data = np.load('../res/coulomb/CO2.npy')

    N = data.shape[0]
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    c = md.coulomb.ewald.EwaldOrthoganal(domain=A.domain, real_cutoff=rc, alpha=alpha, shared_memory=True)
    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"


    A.positions = PositionDat(ncomp=3)
    A.forces = ParticleDat(ncomp=3)
    A.charges = ParticleDat(ncomp=1)

    energy = GlobalArray(size=1, dtype=ctypes.c_double, shared_memory=False)

    if mpi_rank == 0:
        A.positions[:] = data[:,0:3:]
        A.charges[:, 0] = data[:,3]
    A.scatter_data_from(0)

    c.evaluate_contributions(positions=A.positions, charges=A.charges)

    energy[0] = 0.0
    c.extract_forces_energy(A.positions, A.charges, A.forces, energy)

    assert abs(energy[0]*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "{}, {}".format(energy[0]*c.internal_to_ev(), energy[0])

