__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"


import numpy as np
import pytest
from decimal import Decimal
import ppmd as md

import scipy
import scipy.constants

from math import sqrt

mpi_rank = md.mpi.MPI.COMM_WORLD.Get_rank()
mpi_size = md.mpi.MPI.COMM_WORLD.Get_size()
ParticleDat = md.data.ParticleDat


# this test is fairly pointless as the result is some tiny number
@pytest.mark.skip
def test_ewald_energy_python_nacl_1():
    """
    Test that the python implementation of ewald calculates the correct 
    real space contribution and self interaction contribution.
    """

    if mpi_rank > 0:
        return

    eta = 0.26506
    alpha = eta**2.
    rc = 12.

    e = 30.0
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.CoulombicEnergy(domain=domain, real_cutoff=rc, alpha=alpha)

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"

    data = np.load('../res/coulomb/NACL.npy')

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.test_evaluate_python_lr(positions=positions, charges=charges)

    print rs[0]*c.internal_to_ev(), rs[1]*c.internal_to_ev()
    assert abs(rs[0]*c.internal_to_ev() - 0.5223894616E-26) < 10.**-3, "Energy from loop back over particles"
    assert abs(rs[1]*c.internal_to_ev() - 0.5223894616E-26) < 10.**-3, "Energy from structure factor"

#@pytest.mark.skip
def test_ewald_energy_python_co2_1():
    """
    Test that the python implementation of ewald calculates the correct 
    real space contribution and self interaction contribution.
    """

    if mpi_rank > 0:
        return

    eta = 0.26506
    alpha = eta**2.
    rc = 12.

    e = 24.47507
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.CoulombicEnergy(domain=domain, real_cutoff=rc, alpha=alpha)

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"

    data = np.load('../res/coulomb/CO2.npy')

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    #positions[:, 0] -= e*0.5
    #positions[:, 1] -= e*0.5
    #positions[:, 2] -= e*0.5
    #print(np.max(positions[:,0]), np.min(positions[:,0]))
    #print(np.max(positions[:,1]), np.min(positions[:,1]))
    #print(np.max(positions[:,2]), np.min(positions[:,2]))

    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.test_evaluate_python_lr(positions=positions, charges=charges)

    assert abs(rs[1]*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "Energy from structure factor"
    assert abs(rs[0]*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "Energy from loop back over particles"

@pytest.mark.skip
def test_ewald_energy_python_co2_2():
    """
    Test that the python implementation of ewald calculates the correct 
    real space contribution and self interaction contribution.
    """

    if mpi_rank > 0:
        return

    eta = 0.26506
    alpha = eta**2.
    rc = 12.
    e0 = 30.
    e1 = 40.
    e2 = 50.

    domain = md.domain.BaseDomainHalo(extent=(e0,e1,e2))
    c = md.coulomb.CoulombicEnergy(
        domain=domain,
        real_cutoff=12.,
        alpha=alpha,
        recip_cutoff=0.2667*scipy.constants.pi*2.0,
        recip_nmax=(8,11,14)
    )

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"


    data = np.load('../res/coulomb/CO2cuboid.npy')


    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]

    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.test_evaluate_python_lr(positions=positions, charges=charges)
    assert abs(rs[0]*c.internal_to_ev() - 0.3063162184E+02) < 10.**-3, "Energy from loop back over particles"
    assert abs(rs[1]*c.internal_to_ev() - 0.3063162184E+02) < 10.**-3, "Energy from structure factor"











