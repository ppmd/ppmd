"""
This example demonstrates the initialisation of a "State" with: domain
boundary condition, number of particles and ParticleDat data structures. 

The ParticleDat data structures are initialised with data which is
subsequently broadcast across MPI ranks.

The example is ran by executing:

    mpirun -n 8 python 00_init_data_scatter.py

from a shell or jobscript. An example output is given at the end of
this file.
"""

import ctypes
from ppmd import *

# sys is imported purely to flush stdout
import sys


# define constants for the simulation
N = 3*4*5
E = (3.0, 4.0, 5.0)

# minimum required setup of a "state" named A
A = state.State()
# Set the number of particles
A.npart = N
# create a domain with extent E and periodic boundary conditions
A.domain = domain.BaseDomainHalo(extent=E)
A.domain.boundary_condition = domain.BoundaryTypePeriodic()

# Add the data required per particle to the state A
# A single "PositionDat must exist for parallel decomposition
# The names A.* are user choice, names beginning with an underscore
# are reserved.

# Positions with 3 components, default datatype is ctypes.c_double
A.pos = data.PositionDat(ncomp=3)
# Storage of 1 ctypes.c_long per particle.
A.id = data.ParticleDat(ncomp=1, dtype=ctypes.c_long)

# In this example we initialise the data structures with values on 
# rank 0 then scatter the data across mpi ranks
mpi_rank = mpi.MPI.COMM_WORLD.Get_rank()
if mpi_rank == 0:
    # use a utility function to create a cubic lattice of points
    # and use these points for positions
    A.pos[:] = utility.lattice.cubic_lattice(n=(3,4,5), e=E)
    # initialise the ids with [0, N-1]
    A.id[:, 0] = range(N)

# scatter the data from rank 0 across all mpi ranks
A.scatter_data_from(0)

# print data on each rank
mpi_size = mpi.MPI.COMM_WORLD.Get_size()
for rx in range(mpi_size):
    if mpi_rank == rx:
        print(8*"-", "rank:", mpi_rank, 8*"-")

        # The number of particles stored on an MPI rank is given by the value
        # of A.npart_local
        print("\tNumber of particles on rank:", A.npart_local)
        for px in range(A.npart_local):
            print("\tid: {: 6d},\tpos: {: 8.4f} {: 8.4f} {: 8.4f}".format(
                A.id[px, 0], A.pos[px, 0], A.pos[px, 1], A.pos[px, 2]))
    sys.stdout.flush()
    mpi.MPI.COMM_WORLD.Barrier()
mpi.MPI.COMM_WORLD.Barrier()



"""
-------- rank: 0 --------
	Number of particles on rank: 4
	id:      0,	pos:  -1.0000  -1.5000  -2.0000
	id:      1,	pos:  -1.0000  -1.5000  -1.0000
	id:      6,	pos:  -1.0000  -0.5000  -1.0000
	id:      5,	pos:  -1.0000  -0.5000  -2.0000
-------- rank: 1 --------
	Number of particles on rank: 8
	id:     46,	pos:   1.0000  -0.5000  -1.0000
	id:     45,	pos:   1.0000  -0.5000  -2.0000
	id:     41,	pos:   1.0000  -1.5000  -1.0000
	id:     40,	pos:   1.0000  -1.5000  -2.0000
	id:     26,	pos:   0.0000  -0.5000  -1.0000
	id:     25,	pos:   0.0000  -0.5000  -2.0000
	id:     21,	pos:   0.0000  -1.5000  -1.0000
	id:     20,	pos:   0.0000  -1.5000  -2.0000
-------- rank: 2 --------
	Number of particles on rank: 4
	id:     16,	pos:  -1.0000   1.5000  -1.0000
	id:     15,	pos:  -1.0000   1.5000  -2.0000
	id:     11,	pos:  -1.0000   0.5000  -1.0000
	id:     10,	pos:  -1.0000   0.5000  -2.0000
-------- rank: 3 --------
	Number of particles on rank: 8
	id:     56,	pos:   1.0000   1.5000  -1.0000
	id:     55,	pos:   1.0000   1.5000  -2.0000
	id:     51,	pos:   1.0000   0.5000  -1.0000
	id:     50,	pos:   1.0000   0.5000  -2.0000
	id:     36,	pos:   0.0000   1.5000  -1.0000
	id:     35,	pos:   0.0000   1.5000  -2.0000
	id:     31,	pos:   0.0000   0.5000  -1.0000
	id:     30,	pos:   0.0000   0.5000  -2.0000
-------- rank: 4 --------
	Number of particles on rank: 6
	id:      9,	pos:  -1.0000  -0.5000   2.0000
	id:      8,	pos:  -1.0000  -0.5000   1.0000
	id:      2,	pos:  -1.0000  -1.5000   0.0000
	id:      3,	pos:  -1.0000  -1.5000   1.0000
	id:      4,	pos:  -1.0000  -1.5000   2.0000
	id:      7,	pos:  -1.0000  -0.5000   0.0000
-------- rank: 5 --------
	Number of particles on rank: 12
	id:     49,	pos:   1.0000  -0.5000   2.0000
	id:     48,	pos:   1.0000  -0.5000   1.0000
	id:     47,	pos:   1.0000  -0.5000   0.0000
	id:     44,	pos:   1.0000  -1.5000   2.0000
	id:     43,	pos:   1.0000  -1.5000   1.0000
	id:     42,	pos:   1.0000  -1.5000   0.0000
	id:     29,	pos:   0.0000  -0.5000   2.0000
	id:     28,	pos:   0.0000  -0.5000   1.0000
	id:     27,	pos:   0.0000  -0.5000   0.0000
	id:     24,	pos:   0.0000  -1.5000   2.0000
	id:     23,	pos:   0.0000  -1.5000   1.0000
	id:     22,	pos:   0.0000  -1.5000   0.0000
-------- rank: 6 --------
	Number of particles on rank: 6
	id:     19,	pos:  -1.0000   1.5000   2.0000
	id:     18,	pos:  -1.0000   1.5000   1.0000
	id:     17,	pos:  -1.0000   1.5000   0.0000
	id:     14,	pos:  -1.0000   0.5000   2.0000
	id:     13,	pos:  -1.0000   0.5000   1.0000
-------- rank: 7 --------
	Number of particles on rank: 12
	id:     12,	pos:  -1.0000   0.5000   0.0000
	id:     59,	pos:   1.0000   1.5000   2.0000
	id:     58,	pos:   1.0000   1.5000   1.0000
	id:     57,	pos:   1.0000   1.5000   0.0000
	id:     54,	pos:   1.0000   0.5000   2.0000
	id:     53,	pos:   1.0000   0.5000   1.0000
	id:     52,	pos:   1.0000   0.5000   0.0000
	id:     39,	pos:   0.0000   1.5000   2.0000
	id:     38,	pos:   0.0000   1.5000   1.0000
	id:     37,	pos:   0.0000   1.5000   0.0000
	id:     34,	pos:   0.0000   0.5000   2.0000
	id:     33,	pos:   0.0000   0.5000   1.0000
	id:     32,	pos:   0.0000   0.5000   0.0000
"""

