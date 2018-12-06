from ppmd import *

import numpy as np
from ctypes import *
from ppmd import *

State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
Kernel = kernel.Kernel
GlobalArray = data.GlobalArray
ParticleLoop = loop.ParticleLoopOMP

def test_basic_existance():
    Nx = 3
    Ny = 4
    Nz = 5
    rc = 1.0 + 10.**-8
    rn = rc * (1.1)

    extent = (float(Nx), float(Ny), float(Nz))
    
    assert Ny % 2 == 0
    N = Nx * Ny * Nz
    NA = Nx * (Ny // 2) * Nz
    NB = Nx * (Ny // 2) * Nz

    cuboid = domain.BaseDomainHalo(extent)
    cuboid.boundary_condition = domain.BoundaryTypePeriodic()
    
    A = State()
    B = State()

    A.domain = cuboid
    B.domain = cuboid

    A.npart = NA
    B.npart = NB

    A.positions = PositionDat()
    B.positions = PositionDat()

    A.ncount = ParticleDat(ncomp=1, dtype=c_int)
    B.ncount = ParticleDat(ncomp=1, dtype=c_int)

    A.gcount = GlobalArray(ncomp=1, dtype=c_int)
    B.gcount = GlobalArray(ncomp=1, dtype=c_int)

    cubic_lattice = utility.lattice.cubic_lattice((Nx, Ny, Nz), extent)

    A.positions[:, :] = cubic_lattice[0:N:2, :]
    B.positions[:, :] = cubic_lattice[1:N:2, :]

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    count_kernel_src = r'''
    C.i[0]++;
    GC[0]++;
    '''

    count_kernel = Kernel('count_kernel', count_kernel_src)
    count_loop_A = ParticleLoop(count_kernel, {'C': A.ncount(access.INC_ZERO), 'GC': A.gcount(access.INC_ZERO)})
    count_loop_B = ParticleLoop(count_kernel, {'C': B.ncount(access.INC_ZERO), 'GC': B.gcount(access.INC_ZERO)})
    
    count_loop_A.execute()
    count_loop_B.execute()

    assert A.gcount[0] == NA
    assert B.gcount[0] == NB

    for px in range(A.npart_local):
        assert A.ncount[px, 0] == 1

    for px in range(B.npart_local):
        assert B.ncount[px, 0] == 1    






