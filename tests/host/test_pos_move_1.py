import pytest

from ppmd import mpi
from ppmd.state import State
from ppmd.domain import BaseDomainHalo, BoundaryTypePeriodic
from ppmd.data import PositionDat, ParticleDat
from ppmd.coulomb.ewald_half import EwaldOrthoganalHalf
from ppmd.coulomb.fmm import PyFMM

from math import sqrt
import numpy as np

mpi_rank = mpi.MPI.COMM_WORLD.Get_rank()
mpi_size = mpi.MPI.COMM_WORLD.Get_size()
MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier
DEBUG = False

def compute_dipole(positions, charges):
    di = np.zeros(3)
    for cx in range(3):
        di[cx] = np.sum(positions[:, cx]*charges[:].flatten())
    return di


@pytest.mark.skipif("mpi_size > 1")
def test_new_position_values_1():
    
    N1 = 12
    N = 8 * N1
    E1 = 4.
    E = 2 * E1

    A = State()
    A.npart = N
    A.domain = BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()

    A.pos = PositionDat(ncomp=3)
    A.chr = ParticleDat(ncomp=1)
    A.frc = ParticleDat(ncomp=3)

    rng = np.random.RandomState(seed=34)
    
    offsets = (
        ( 0.5*E1, 0.5*E1, 0.5*E1),
        (-0.5*E1, 0.5*E1, 0.5*E1),
        (-0.5*E1,-0.5*E1, 0.5*E1),
        ( 0.5*E1,-0.5*E1, 0.5*E1),
        ( 0.5*E1, 0.5*E1,-0.5*E1),
        (-0.5*E1, 0.5*E1,-0.5*E1),
        (-0.5*E1,-0.5*E1,-0.5*E1),
        ( 0.5*E1,-0.5*E1,-0.5*E1)
    )

    fact = (
        ( 1., 1., 1.),
        (-1., 1., 1.),
        (-1.,-1., 1.),
        ( 1.,-1., 1.),
        ( 1., 1.,-1.),
        (-1., 1.,-1.),
        (-1.,-1.,-1.),
        ( 1.,-1.,-1.)
    )
    
    def remove_dipole(pos, cha):
        new_pos = np.zeros((N, 3), dtype='float')
        new_chr = np.zeros((N, 1), dtype='float')
        for qx in range(8):
            for px in range(N1):
                new_pos[qx*N1+px, :] = pos[px, :] * fact[qx]
                new_chr[qx*N1+px, :] = cha[px]
        for qx in range(8):
            for px in range(N1):
                new_pos[qx*N1+px, :] += offsets[qx]
        
        return new_pos, new_chr
    
    # make some initial dipole free config (gets discarded)
    pos_tmp = rng.uniform(low=-0.5 * E1, high=0.5 * E1, size=(N1, 3))
    chr_tmp = np.zeros(N1, dtype='float')
    for px in range(N1):
        chr_tmp[px] = (-1.)**px
    bias = float(np.sum(chr_tmp)) / N1
    chr_tmp[:] -= bias

    p, q = remove_dipole(pos_tmp, chr_tmp)
    A.pos[:] = p
    A.chr[:] = q

    A.scatter_data_from(0)
    
    # create an fmm, ewald instance
    fmm = PyFMM(domain=A.domain, r=3, l=10, free_space=False)
    ewald = EwaldOrthoganalHalf(
        domain=A.domain,
        real_cutoff=E * 0.2,
        eps=10.**-6,
        shared_memory='omp'
    )
    

    for testx in range(20):
        # create some new particle positions
        pos_tmp = rng.uniform(low=-0.5 * E1, high=0.5 * E1, size=(N1, 3))

        p, q = remove_dipole(pos_tmp, chr_tmp)
        
        A.pos[:A.npart_local:, :] = p
        A.chr[:A.npart_local:, :] = q
        
        # check config is sensible
        dipole = compute_dipole(A.pos[:N:, :], A.chr[:N:, 0])
        assert abs(dipole[0]) < 10.**-10
        assert abs(dipole[1]) < 10.**-10
        assert abs(dipole[2]) < 10.**-10
        assert abs(np.sum(A.chr[:N:, 0])) < 10.**-10

        fmm_phi = fmm(A.pos, A.chr)
        ewald_phi = ewald(A.pos, A.chr, forces=A.frc)

        assert abs( (fmm_phi - ewald_phi) / abs(ewald_phi) ) < 10.**-4


    fmm.free()















