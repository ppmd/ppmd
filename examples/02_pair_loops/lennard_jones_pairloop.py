# Lennard-Jones example to compute forces and system potential energy


import numpy as np
from ctypes import *
from ppmd import *

# some alias for readability and easy modification if we ever
# wish to use CUDA.

PairLoop = pairloop.CellByCellOMP
ParticleLoop = loop.ParticleLoopOMP
State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
Kernel = kernel.Kernel
GlobalArray = data.GlobalArray
Constant = kernel.Constant

# Some parameters

N = 100
E = 10.0

# PPMD must be able to decompose the domain into cells with at least 3 cells per
# dimension
r_c = E/6.

# Lennard Jones parameters
epsilon = 1.0
sigma = 1.0

# in MD simulations it is common to use neighbour lists or cell decomposions
# built with a cutoff larger than the interaction cutoff.
r_n = r_c + 0.1*r_c

# make a state object and set the global number of particles N
A = State()
A.npart = N

# give the state a domain and boundary condition
A.domain = domain.BaseDomainHalo(extent=(E, E, E))
A.domain.boundary_condition = domain.BoundaryTypePeriodic()

# add a PositionDat to contain positions
A.pos = PositionDat(ncomp=3)
A.force = ParticleDat(ncomp=3)



# set some random positions, origin is in the centre
rng = np.random.RandomState(512)
A.pos[:] = rng.uniform(low=-.5*E, high=.5*E, size=(N, 3))
A.force[:] = 0.0

# system energy store
lj_energy = GlobalArray(ncomp=1, dtype=c_double)


# broadcast the data accross MPI ranks
A.scatter_data_from(0)


# kernel constants
constants = (
    Constant('CV', 4. * epsilon),
    Constant('CF', -48 * epsilon / sigma ** 2),
    Constant('sigma2', sigma ** 2),
    Constant('rc2', r_c ** 2),
    Constant('internalshift', (sigma / r_c) ** 6.0 - (sigma / r_c) ** 12.0)
)


# the pairloop guarantees that all particles such that |r_i - r_j| < r_n
# are looped over. It may also propose pairs of particles such that
# |r_i - r_j| >= r_n and it is the users responsibility to mask off these
# cases
kernel_src = '''
// Vector displacement from particle i to particle j.
const double R0 = P.j[0] - P.i[0];
const double R1 = P.j[1] - P.i[1];
const double R2 = P.j[2] - P.i[2];

// distance squared
const double r2 = R0*R0 + R1*R1 + R2*R2;

// (sigma/r)**2, (sigma/r)**4 and  (sigma/r)**6
const double r_m2 = sigma2/r2;
const double r_m4 = r_m2*r_m2;
const double r_m6 = r_m4*r_m2;

// increment global energy with this interaction
U[0] += (r2 < rc2) ? 0.5*CV*((r_m6-1.0)*r_m6 + internalshift) : 0.0;

// (sigma/r)**8
const double r_m8 = r_m4*r_m4;

// compute force magnitude
const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

// increment force on particle i
F.i[0] += (r2 < rc2) ? f_tmp*R0 : 0.0;
F.i[1] += (r2 < rc2) ? f_tmp*R1 : 0.0;
F.i[2] += (r2 < rc2) ? f_tmp*R2 : 0.0;
'''

lj_kernel = Kernel('LJ-12-6', kernel_src, constants)

# create a pairloop
lj_pairloop = PairLoop(
    lj_kernel, 
    {
        'P': A.pos(access.READ),
        'F': A.force(access.INC_ZERO),
        'U': lj_energy(access.INC_ZERO)
    },
    shell_cutoff=r_n
)

# launch the kernel
lj_pairloop.execute()

if mpi.MPI.COMM_WORLD.rank == 0:
    print("system energy:", lj_energy[0])
    print("First few forces on rank 0:")
    print("\t", A.force[0, :])
    print("\t", A.force[1, :])
    print("\t", A.force[2, :])











