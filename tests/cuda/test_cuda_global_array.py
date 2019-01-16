import math
import pytest
import ctypes
import numpy as np

import ppmd as md
import ppmd.cuda as mdc
from ppmd.access import *

cuda = pytest.mark.skipif("mdc.CUDA_IMPORT is False")

print(mdc.CUDA_IMPORT_ERROR)

GlobalArray = mdc.cuda_data.GlobalArray
ParticleLoop = mdc.cuda_loop.ParticleLoop
PairLoop = mdc.cuda_pairloop.PairLoopNeighbourListNS
State = mdc.cuda_state.State
PositionDat = mdc.cuda_data.PositionDat
ParticleDat = mdc.cuda_data.ParticleDat

@cuda
def test_ga_1():
    nc = 2
    ga1 = GlobalArray(ncomp=nc, dtype=ctypes.c_int)
    ga1.set(1)
    for px in range(nc):
        assert ga1[px] == 1
    
    
@cuda
def test_ga_2():

    nc = 2
    N = 10
    ga1 = GlobalArray(ncomp=nc, dtype=ctypes.c_int)
    ga1.set(1)

    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(1.,1.,1.,))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    
    A.pos = PositionDat(ncomp=3)
    A.gac = ParticleDat(ncomp=nc, dtype=ctypes.c_int)
    A.pos[:] = np.random.uniform(-0.5, 0.5, (N,3))
    A.gac[:] = 0
    A.scatter_data_from(0)

    kernel = md.kernel.Kernel(
        'cuda_test_ga_2',
        """
        gac.i[0] = GA[0];
        gac.i[1] = GA[1];
        """
    )
    loop = ParticleLoop(kernel,
        {
           'GA': ga1(md.access.READ),
           'gac': A.gac(md.access.INC)
        }
    )
    loop.execute()

    for px in range(A.npart_local):
        for nx in range(nc):
            assert A.gac[px, nx] == 1
    
    A.gac[:,0] = 1
    A.gac[:,1] = 2

    kernel = md.kernel.Kernel(
        'cuda_test_ga_3',
        """
        GA[0] += gac.i[0]; 
        GA[1] += gac.i[1]; 
        """
    )
    loop2 = ParticleLoop(kernel,
        {
           'GA': ga1(md.access.INC_ZERO),
           'gac': A.gac(md.access.READ)
        }
    )
    loop2.execute()
    
    for nx in range(nc):
        assert ga1[nx] == N * (nx + 1)
    
    ga1.set(1000000)

    loop2.execute()
    for nx in range(nc):
        assert ga1[nx] == N * (nx + 1)

    loop3 = ParticleLoop(kernel,
        {
           'GA': ga1(md.access.INC),
           'gac': A.gac(md.access.READ)
        }
    )
    loop3.execute()

    for nx in range(nc):
        assert ga1[nx] == 2 * N * (nx + 1)

@cuda
def test_ga_pair_loop_NS_FCC_1():
    """
    Set a cutoff slightly larger than the 3rd nearest neighbour distance in
    the grid
    """

    N = 1000
    crN = 10 #cubert(N)
    E = 8.

    Eo2 = E/2.

    tol = 10.**(-12)


    A = State()

    crN2 = 10

    A.npart = (crN2**3)*4

    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    A.P = PositionDat(ncomp=3)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.nn = GlobalArray(ncomp=1, dtype=ctypes.c_int)
    A.nn2 = GlobalArray(ncomp=2, dtype=ctypes.c_int)
    A.nset = GlobalArray(ncomp=1, dtype=ctypes.c_int)
    A.nset.set(4)

    cell_width = (0.5*float(E))/float(crN2)

    A.P[:] = md.utility.lattice.fcc((crN2, crN2, crN2), (E, E, E))

    A.npart_local = (crN2**3)*4
    NTOTAL = (crN2**3)*4
    A.filter_on_domain_boundary()

    kernel_code = '''
    NC.i[0]+=1;
    NN[0]++;
    NN2[0]+=NSET[0];
    NN2[1]+=2;
    '''

    kernel = md.kernel.Kernel('test_ga_3',code=kernel_code)
    kernel_map = {'P': A.P(md.access.R),
                  'NC': A.nc(md.access.INC_ZERO),
                  'NN': A.nn(md.access.INC_ZERO),
                  'NSET': A.nset(md.access.READ),
                  'NN2': A.nn2(md.access.INC_ZERO)}

    loop = PairLoop(
        kernel=kernel,
        dat_dict=kernel_map,
        shell_cutoff=math.sqrt(2.)*cell_width+tol
    )

    loop.execute()
    for ix in range(A.npart_local):
        assert A.nc[ix] == 12

    assert A.nn[0] == 12*NTOTAL
    assert A.nn2[0] == 12*NTOTAL*4
    assert A.nn2[1] == 12*NTOTAL*2


