import numpy as np
import ctypes

from ppmd import *
from ppmd.cuda import *



# n=25 reasonable size
n = 100
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

COM = cuda_cell.CellOccupancyMatrix()

# Create a particle dat with the positions in from the sim1.state
sim1.state.d_positions = cuda_data.ParticleDat(initial_value=sim1.state.positions.dat)


COM.setup(sim1.state.as_func('n'), sim1.state.d_positions, sim1.state.domain)

COM.sort()

# Create a host.Array to store the computed cell occupancy matrix.
_s = COM.matrix.ncol*COM.matrix.nrow
gpu_occ = host.Array(ncomp=_s, dtype=ctypes.c_int)


# Copy from device to host.
cuda_runtime.cuda_mem_cpy(gpu_occ.ctypes_data, COM.matrix.ctypes_data, ctypes.c_size_t(_s * ctypes.sizeof(ctypes.c_int)), 'cudaMemcpyDeviceToHost')

cell.cell_list.sort()
# alias the cell list
_cl = cell.cell_list.cell_list


_n = 0

for _c in range(sim1.state.domain.cell_count):

    ix = _cl[_cl.end - sim1.state.domain.cell_count + _c]

    host_cl = []


    while ix > -1:
        
        host_cl.append(ix)
        
        ix = _cl[ix]

    gpu_cl = []
    for ix in range(len(host_cl)):
        gpu_cl.append(gpu_occ.dat[COM.layers_per_cell()*_c+ix])



    host_cl.sort()
    gpu_cl.sort()

    _s = True

    for ix in range(len(host_cl)):
        _s = _s and (host_cl == gpu_cl)
        _n += 1



_s2 = (_n == N)
print "Lists same:", _s, ", Num particles checked is same as total num particles:", _s2

print "Test passed:", _s and _s2







