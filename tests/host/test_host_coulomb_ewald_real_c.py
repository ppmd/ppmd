__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import numpy as np
import scipy
import scipy.constants
from math import pi

import ctypes
import ppmd as md


import os
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'), filename)


mpi_rank = md.mpi.MPI.COMM_WORLD.Get_rank()
mpi_ncomp = md.mpi.MPI.COMM_WORLD.Get_size()
ParticleDat = md.data.ParticleDat
PositionDat = md.data.PositionDat
ScalarArray = md.data.ScalarArray
State = md.state.BaseMDState


def test_ewald_energy_python_nacl_c_1():

    SHARED_MEMORY = False

    eta = 0.26506
    alpha = eta**2.
    rc = 12.
    e0 = 30.
    e1 = 30.
    e2 = 30.
    eo2 = 15.

    data = np.load(get_res_file_path('coulomb/NACL.npy'))

    data[:,:3] -= eo2

    N = data.shape[0]
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(e0,e1,e2))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    c = md.coulomb.ewald.EwaldOrthoganal(
        domain=A.domain,
        real_cutoff=rc,
        alpha=alpha,
        recip_cutoff=0.28601*pi*2.0,
        recip_nmax=(9,9,9),
        shared_memory=SHARED_MEMORY
    )

    A.positions = PositionDat(ncomp=3)
    A.forces = ParticleDat(ncomp=3)
    A.charges = ParticleDat(ncomp=1)

    energy = ScalarArray(ncomp=1, dtype=ctypes.c_double)
    energy_real = ScalarArray(ncomp=1, dtype=ctypes.c_double)
    energy_self = ScalarArray(ncomp=1, dtype=ctypes.c_double)

    if mpi_rank == 0:
        A.positions[:] = data[:,0:3:]
        A.charges[:, 0] = data[:,3]

    A.scatter_data_from(0)

    c.evaluate_contributions(positions=A.positions, charges=A.charges)

    c.extract_forces_energy_reciprocal(A.positions, A.charges, A.forces, energy)

    rs = c._test_python_structure_factor()

    assert abs(rs*c.internal_to_ev() - 0.5223894616E-26) < 10.**-3, "structure factor"
    assert abs(energy[0]*c.internal_to_ev() - 0.5223894616E-26) < 10.**-3, "particle loop"

    c.extract_forces_energy_real(A.positions, A.charges, A.forces, energy_real)

    c.evaluate_self_interactions(A.charges, energy_self)

    assert abs(energy_real[0]*c.internal_to_ev() + energy_self[0]*c.internal_to_ev() + 0.4194069853E+04) < 10.**-2, "bad real space part"








