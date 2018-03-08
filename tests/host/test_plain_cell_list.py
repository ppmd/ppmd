
from ppmd import *
import numpy as np
import sys
np.set_printoptions(threshold=np.nan)

MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
BARRIER = mpi.MPI.COMM_WORLD.Barrier

def test_init_1():
    
    extent = (4., 4., 4.)
    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=extent)
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    

    N = 1000
    A.npart = N
    A.P = data.PositionDat(ncomp=3)


    rng = np.random.RandomState(seed=567)
    A.P[:,0] = rng.uniform(low=-0.5*extent[0], high=0.5*extent[0], size=N)
    A.P[:,1] = rng.uniform(low=-0.5*extent[1], high=0.5*extent[1], size=N)
    A.P[:,2] = rng.uniform(low=-0.5*extent[2], high=0.5*extent[2], size=N)

    A.scatter_data_from(0)

    large_width = 2.0

    A.cell_decompose(large_width)
    A.get_cell_to_particle_map().check()
    A.P.halo_exchange()
    
    nwidth = 1.0

    cl = plain_cell_list.PlainCellList(nwidth, A.domain.boundary)
    
    
    if MPIRANK == 0:
        print("\n")

    for rx in range(MPISIZE):
        if rx == MPIRANK:
            print(MPIRANK, A.domain.boundary, cl.cell_array)
            print("--->", A.get_cell_to_particle_map().cell_contents_count[:])
            sys.stdout.flush()
        BARRIER()
    
    if MPIRANK == 0:
        print(60*'-')

    cl.sort(A.P, A.npart_local + A.npart_halo)

    
    ca = cl.cell_array
    s = ca[0]*ca[1]

    for rx in range(MPISIZE):
        if rx == MPIRANK:
            print(cl.cell_array)
            for zz in range(ca[2]):
                print("--->\n", cl.cell_contents_count[zz*s:(zz+1)*s:].reshape(ca[1], ca[0]))
            sys.stdout.flush()
        BARRIER()
    
