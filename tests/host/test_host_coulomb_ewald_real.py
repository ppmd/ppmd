from __future__ import print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import numpy as np
import scipy
import scipy.constants
from math import pi
import pytest

import ppmd as md


import os
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'), filename)


mpi_rank = md.mpi.MPI.COMM_WORLD.Get_rank()
mpi_size = md.mpi.MPI.COMM_WORLD.Get_size()
ParticleDat = md.data.ParticleDat
PositionDat = md.data.PositionDat
ScalarArray = md.data.ScalarArray
State = md.state.BaseMDState
GlobalArray = md.data.GlobalArray

@pytest.mark.skipif(mpi_size>1, reason="MPI::Get_size()>1")
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
    c = md.coulomb.ewald.EwaldOrthoganal(domain=domain, real_cutoff=rc, alpha=alpha)

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"

    data = np.load(get_res_file_path('coulomb/NACL.npy'))

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.test_evaluate_python_sr(positions=positions, charges=charges)
    selfinteraction = c.test_evaluate_python_self(charges)

    localsr = rs * c.internal_to_ev()
    selfij = selfinteraction * c.internal_to_ev()

    # the tolerance here is about 6 decimal places
    assert abs(selfij + localsr + 0.4194069853E+04)< 10.**-2, "real + self error"

@pytest.mark.skipif(mpi_size>1, reason="MPI::Get_size()>1")
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
    c = md.coulomb.ewald.EwaldOrthoganal(domain=domain, real_cutoff=rc, alpha=alpha)

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"

    data = np.load(get_res_file_path('coulomb/CO2.npy'))

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.test_evaluate_python_sr(positions=positions, charges=charges)
    selfinteraction = c.test_evaluate_python_self(charges)

    localsr = rs * c.internal_to_ev()
    selfij = selfinteraction * c.internal_to_ev()


    # the tolerance here is about 6 decimal places
    assert abs(selfij + localsr + 0.110417414E5)< 10.**-2, "real + self error"


@pytest.mark.skipif(mpi_size>1, reason="MPI::Get_size()>1")
def test_ewald_energy_python_nacl_2():
    """
    Test that the python implementation of ewald sets up the Ewald calculation
    consistently.
    """

    if mpi_rank > 0:
        return

    e = 30.0
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.ewald.EwaldOrthoganal(domain=domain, real_cutoff=12.)
    assert abs(c.recip_cutoff - 0.28601*scipy.constants.pi*2.0) < 10.**-1, "recip space cutoff"
    assert abs(c.real_cutoff - 12.) < 10.**-15., "real space cutoff"
    assert c.kmax[0] == 9, "kmax_x"
    assert c.kmax[1] == 9, "kmax_y"
    assert c.kmax[2] == 9, "kmax_z"
    assert abs(c.recip_vectors[0][0] - 0.0333333*2.*scipy.constants.pi) < 10.**-5, "xrecip vector"
    assert abs(c.recip_vectors[1][1] - 0.0333333*2.*scipy.constants.pi) < 10.**-5, "yrecip vector"
    assert abs(c.recip_vectors[2][2] - 0.0333333*2.*scipy.constants.pi) < 10.**-5, "zrecip vector"


@pytest.mark.skipif(mpi_size>1, reason="MPI::Get_size()>1")
def test_ewald_energy_python_co2_2():
    """
    Test that the python implementation of ewald sets up the Ewald calculation
    consistently.
    """

    if mpi_rank > 0:
        return

    e = 24.4750735
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.ewald.EwaldOrthoganal(domain=domain, real_cutoff=12.)
    assert abs(c.recip_cutoff - 0.28601*scipy.constants.pi*2.0) < 10.**-1, "recip space cutoff"
    assert abs(c.real_cutoff - 12.) < 10.**-15., "real space cutoff"
    assert c.kmax[0] == 7, "kmax_x"
    assert c.kmax[1] == 7, "kmax_y"
    assert c.kmax[2] == 7, "kmax_z"
    assert abs(c.recip_vectors[0][0] - 0.0408579*2.*scipy.constants.pi) < 10.**-5, "xrecip vector"
    assert abs(c.recip_vectors[1][1] - 0.0408579*2.*scipy.constants.pi) < 10.**-5, "yrecip vector"
    assert abs(c.recip_vectors[2][2] - 0.0408579*2.*scipy.constants.pi) < 10.**-5, "zrecip vector"



@pytest.mark.skipif(mpi_size>1, reason="MPI::Get_size()>1")
def test_ewald_energy_python_co2_3():
    """
    Test that the python implementation of ewald sets up the Ewald calculation
    consistently.
    """

    if mpi_rank > 0:
        return

    e0 = 30.
    e1 = 40.
    e2 = 50.

    domain = md.domain.BaseDomainHalo(extent=(e0,e1,e2))
    c = md.coulomb.ewald.EwaldOrthoganal(
        domain=domain,
        real_cutoff=12.,
        alpha=0.26506**2.,
        recip_cutoff=0.2667*scipy.constants.pi*2.0,
        recip_nmax=(8,11,14)
    )
    assert abs(c.recip_cutoff - 0.2667*scipy.constants.pi*2.0) < 10.**-1, "recip space cutoff"
    assert abs(c.real_cutoff - 12.) < 10.**-15., "real space cutoff"
    assert c.kmax[0] == 8, "kmax_x"
    assert c.kmax[1] == 11, "kmax_y"
    assert c.kmax[2] == 14, "kmax_z"
    assert abs(c.recip_vectors[0][0] - 0.033333*2.*scipy.constants.pi) < 10.**-5, "xrecip vector"
    assert abs(c.recip_vectors[1][1] - 0.025*2.*scipy.constants.pi) < 10.**-5, "yrecip vector"
    assert abs(c.recip_vectors[2][2] - 0.02*2.*scipy.constants.pi) < 10.**-5, "zrecip vector"










