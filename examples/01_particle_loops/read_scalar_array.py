# Example to demonstrate reading data from a ScalarArray in a ParticleLoop
# run with:
#   python read_scalar_array.py
# or:
#   mpirun -n 4 python read_scalar_array.py


import numpy as np
from ctypes import *
from ppmd import *

# some alias for readability and easy modification if we ever
# wish to use CUDA.
ParticleLoop = loop.ParticleLoopOMP
State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
Kernel = kernel.Kernel


N = 100
E = 10.0

# make a state object and set the global number of particles N
A = State()
A.npart = N

# give the state a domain and boundary condition
A.domain = domain.BaseDomainHalo(extent=(E, E, E))
A.domain.boundary_condition = domain.BoundaryTypePeriodic()

# add a PositionDat to contain positions
A.pos = PositionDat(ncomp=3)

# add a ParticleDat to hold 2 ints per particle
A.int_prop = ParticleDat(ncomp=2, dtype=c_int)

# create a ScalarArray of 5 ints
int_array = ScalarArray(ncomp=5, dtype=c_int)
int_array[:] = np.random.randint(low=100, high=10000, size=5)

# set some random positions, origin is in the centre
A.pos[:] = np.random.uniform(low=-.5*E, high=.5*E, size=(N, 3))

# set the first int prop to some random int in {0,...,4}
A.int_prop[:,0] = np.random.randint(0,4,N)

# broadcast the data from rank 0 accross all MPI ranks.
# after this call, A.npart_local will hold the number particles owned
# by a MPI rank
A.scatter_data_from(0)

# Make a kernel that copies the int_prop[0]^th element from the 
# ScalarArray into int_prop[1] for each particle.
kernel_src = '''
/* we choose to access the  A.int_prop ParticleDat using the C symbol
  "int_prop" and the ScalarArray int_array on the c symbol "IA". */

int_prop.i[1] = IA[ int_prop.i[0] ];

'''

# make a kernel object with our kernel source, the name is user choice.
copy_kernel = Kernel('particle_loop_scalararray_example', kernel_src)

copy_loop = ParticleLoop(
    copy_kernel, # the kernel to apply to each particle
    {            # map from C symbols to python data with access descriptors
        'int_prop': A.int_prop(access.WRITE),
        'IA': int_array(access.READ)
    }
)


# execute the particleloop
copy_loop.execute()

# use python to check the result is correct
for px in range(A.npart_local):
    assert A.int_prop[px, 1] == int_array[A.int_prop[px, 0]]



