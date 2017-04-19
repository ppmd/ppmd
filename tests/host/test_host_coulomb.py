__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"


import numpy as np
import pytest
from decimal import Decimal
import ppmd as md

mpi_rank = md.mpi.MPI.COMM_WORLD.Get_rank()
mpi_size = md.mpi.MPI.COMM_WORLD.Get_size()
ParticleDat = md.data.ParticleDat

def test_ewald_energy_python():
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

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"


    rs = c.evaluate_python_sr(positions=positions, charges=charges)
    print "self", c.evaluate_python_self(charges=charges)
    print "rs", rs, 'rsdl', '%E' % Decimal(str(rs*9648.530821))
    #print "recip space", c.evaluate_python_lr(positions=positions, charges=charges)




















