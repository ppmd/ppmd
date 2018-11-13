# Example demonstrates writing and executing a pairloop that for each pair of particles
# Evaluates a polynomial P(r) where r<r_c is the inter-particle distance.



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


# Some parameters

N = 100
E = 10.0

# PPMD must be able to decompose the domain into cells with at least 3 cells per
# dimension
r_c = E/6.

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

# set some random positions, origin is in the centre
A.pos[:] = np.random.uniform(low=-.5*E, high=.5*E, size=(N, 3))

# broadcast the data accross MPI ranks
A.scatter_data_from(0)


# GlobalArray to accumalate P(r) evaluations into
poly_eval_sum = GlobalArray(ncomp=1, dtype=c_double)

# polynomial coefficients for a cubic polynomial
poly_coeffs = ScalarArray(ncomp=4, dtype=c_double)
poly_coeffs[:] = np.random.uniform(0, 2, 4)


kernel_src = '''

// compute distance in x, y, and z.
double dx = P.j[0] - P.i[0];
double dy = P.j[1] - P.i[1];
double dz = P.j[2] - P.i[2];

// Compute interparticle distance
double r2 = dx*dx + dy*dy + dz*dz;
double r = sqrt(r2);

// Evaluate the polynomial using Horner's method.
double poly_tmp = ((pc[0]*r + pc[1])*r + pc[2])*r + pc[3];

// Store the result if the two particles were within the cutoff.
pe[0] += (r < {r_c}) ? poly_tmp : 0.0;

'''.format(r_c=r_c)

poly_kernel = Kernel('poly_eval_pairloop', kernel_src)

poly_pairloop = PairLoop(
    poly_kernel, # kernel to apply
    {
        'P':  A.pos(access.READ),
        'pc': poly_coeffs(access.READ),
        'pe': poly_eval_sum(access.INC_ZERO)
    },
    shell_cutoff=r_n
)

poly_pairloop.execute()


print(poly_eval_sum[0])









