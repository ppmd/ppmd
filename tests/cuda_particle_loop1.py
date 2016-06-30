import numpy as np
import ctypes
import math

from ppmd import *
from ppmd.cuda import *

cuda_runtime.DEBUG.level = 0

# n=25 reasonable size
n = 50
N = n**3
# n=860
rho = 1.
mu = 1.0
nsig = 5.0

# Initialise basic domain
test_domain = domain.BaseDomainHalo()

# Initialise LJ potential
test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)

test_pos_init = simulation.PosInitLatticeNRhoRand(N,rho,0.,None)

test_vel_init = simulation.VelInitNormDist(mu,nsig)

test_mass_init = simulation.MassInitIdentical(1.)


sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                                   potential_in=test_potential,
                                   particle_pos_init=test_pos_init,
                                   particle_vel_init=test_vel_init,
                                   particle_mass_init=test_mass_init,
                                   n=N
                                   )


# Create a particle dat with the positions in from the sim1.state
A = cuda_data.ParticleDat(initial_value=sim1.state.positions.data)

C = cuda_data.ParticleDat(ncomp=A.ncomp, npart=A.npart_local, dtype=A.dtype)
# Create a host.Array to store the computed cell occupancy matrix.
h_C = data.ParticleDat(ncomp=C.ncomp, npart=C.npart_local, dtype=C.dtype)


_kernel_code = '''
C[0] = A[0] + B;
C[1] = A[1] + B;
C[2] = A[2] + B;
'''

_kernel = kernel.Kernel('CeApb', _kernel_code, static_args={'B':ctypes.c_double})
_kernel_map = {'A': A(access.R), 'C':C(access.RW)}

_lib = cuda_loop.ParticleLoop(_kernel, _kernel_map)

_lib.execute(n=N, static_args={'B':ctypes.c_double(4.0)})


# Copy from device to host.
cuda_runtime.cuda_mem_cpy(h_C.ctypes_data, C.ctypes_data, ctypes.c_size_t(C.ncomp * C.npart_local * ctypes.sizeof(C.dtype)), 'cudaMemcpyDeviceToHost')

_s = True

for ix in range(N):
    for iy in range(h_C.ncomp):

        _s &= abs(h_C[ix,iy] - (sim1.state.positions.data[ix,iy] + 4.0)) < 10 ** -10

        if not _s:
            print h_C[ix,iy], sim1.state.positions.data[ix,iy] + 4.0, abs(h_C[ix,iy] - (sim1.state.positions.data[ix,iy] + 4.0)), "(ix, iy) =", ix, iy
            print "Test FAILED"
            quit()

if _s:
    print "Test passed."
else:
    print "Test FAILED"








