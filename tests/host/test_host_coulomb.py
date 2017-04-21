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


def test_ewald_energy_python_nacl_1():
    """
    Test that the python implementation of ewald calculates the correct 
    energy
    """

    if mpi_rank > 0:
        return

    eta = 0.26506
    alpha = eta**2.
    rc = 12.

    e = 30.0
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.CoulombicEnergy(domain=domain, real_cutoff=rc, alpha=alpha)

    # !! This is out
    assert abs(c.real_cutoff - 12.) < 1.0, "real space cutoff"

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"


    data = np.load('../res/coulomb/NACL.npy')

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"


    rs = c.evaluate_python_sr(positions=positions, charges=charges)

    epsilon_0 = scipy.constants.epsilon_0
    pi = scipy.constants.pi
    c0 = scipy.constants.physical_constants['atomic unit of charge'][0]
    l0 = 10.**-10

    EkJ =  rs * (1./(4. * pi * epsilon_0 * l0)) * ( c0 ** 2. ) / 1000.

    localsr = 1000. * EkJ / c0
    #print "EeV", localsr

    selfinteraction = np.sum(np.square(charges[:,0]))
    selfinteraction2 = selfinteraction * -1. * sqrt(alpha/pi) / (4.*pi*epsilon_0*l0)
    selfij = selfinteraction2 * (c0**2.) / c0

    #print "self", selfij
    #print "real", selfij + localsr

    print selfij + localsr

    # the tolerance here is about 6 decimal places
    assert abs(selfij + localsr + 0.4194069853E+04)< 10.**-2, "real + self error"




@pytest.mark.skipif('True')
def test_ewald_energy_python_co2_1():
    """
    Test that the python implementation of ewald calculates the correct 
    energy
    """

    if mpi_rank > 0:
        return

    e = 24.4750735
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.CoulombicEnergy(domain=domain, alpha=0.26506)
    assert abs(c.recip_cutoff - 0.28601) < 10.**-5, "recip space cutoff"
    assert abs(c.real_cutoff - 12.) < 1.0, "real space cutoff"
    assert c.kmax[0] == 7, "kmax_x"
    assert c.kmax[1] == 7, "kmax_y"
    assert c.kmax[2] == 7, "kmax_z"
    assert abs(c.recip_vectors[0][0] - 0.0408579) < 10.**-5, "xrecip vector"
    assert abs(c.recip_vectors[1][1] - 0.0408579) < 10.**-5, "yrecip vector"
    assert abs(c.recip_vectors[2][2] - 0.0408579) < 10.**-5, "zrecip vector"

    data = np.load('../res/coulomb/CO2.npy')


    N = data.shape[0]
    N0 = data[data[:, 3] > 0.0].shape[0]
    N1 = data[data[:, 3] < 0.0].shape[0]



    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"


    rs = c.evaluate_python_sr(positions=positions, charges=charges)
    print "self", c.evaluate_python_self(charges=charges)
    print "rs", rs, 'rsdl', '%E' % Decimal(str(rs*9648.530821))
    #print "recip space", c.evaluate_python_lr(positions=positions, charges=charges)


    N0 = data[data[:, 3] > 0.0].shape[0]
    N1 = data[data[:, 3] < 0.0].shape[0]


    Na = scipy.constants.Avogadro
    epsilon_0 = scipy.constants.epsilon_0
    pi = scipy.constants.pi
    c0 = scipy.constants.physical_constants['atomic unit of charge'][0]
    l0 = 10.**-10


    nmol = (N0*28. + N1*16.)/Na
    print "nmol", nmol
    EkJ =  rs * (1./(4. * pi * epsilon_0 * l0)) * ( c0 ** 2. ) / 1000.
    print "EkJ", EkJ
    EkJpmol = EkJ / nmol
    print "kJ mol^-1", EkJpmol




@pytest.mark.skipif('True')
def test_ewald_energy_python_co2_2():
    """
    Test that the python implementation of ewald calculates the correct 
    energy
    """

    if mpi_rank > 0:
        return

    e = 24.4750735
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.CoulombicEnergy(domain=domain, real_cutoff=12.)
    assert abs(c.recip_cutoff - 0.28601) < 10.**-1, "recip space cutoff"
    assert abs(c.real_cutoff - 12.) < 10.**-15., "real space cutoff"
    assert c.kmax[0] == 8, "kmax_x"
    assert c.kmax[1] == 8, "kmax_y"
    assert c.kmax[2] == 8, "kmax_z"
    assert abs(c.recip_vectors[0][0] - 0.0408579) < 10.**-5, "xrecip vector"
    assert abs(c.recip_vectors[1][1] - 0.0408579) < 10.**-5, "yrecip vector"
    assert abs(c.recip_vectors[2][2] - 0.0408579) < 10.**-5, "zrecip vector"

    data = np.load('../res/coulomb/CO2.npy')



    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"


    rs = c.evaluate_python_sr(positions=positions, charges=charges)
    print "self", c.evaluate_python_self(charges=charges)
    print "rs", rs, 'rsdl', '%E' % Decimal(str(rs*9648.530821))
    #print "recip space", c.evaluate_python_lr(positions=positions, charges=charges)




















