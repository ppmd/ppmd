import numpy as np
import ctypes

from ppmd import *
from ppmd.cuda import *



# n=25 reasonable size
n = 4
N = n**3
# n=860
rho = 1.
mu = 1.0
nsig = 5.0

# Initialise basic domain
test_domain = domain.BaseDomainHalo()

# Initialise LJ potential
test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0, rc=0.9)

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
sim1.state.d_positions = cuda_data.ParticleDat(initial_value=sim1.state.positions, name='positions')
print sim1.state.positions.max_npart, sim1.state.positions.npart_local, sim1.state.positions.data

COM.setup(sim1.state.as_func('npart_local'), sim1.state.d_positions, sim1.state.domain)

COM.sort()
cuda_halo.HALOS = cuda_halo.CartesianHalo(COM)


_s = COM.matrix.ncol*COM.matrix.nrow
gpu_occ = host.Array(ncomp=_s, dtype=ctypes.c_int)
cuda_runtime.cuda_mem_cpy(gpu_occ.ctypes_data, COM.matrix.ctypes_data, ctypes.c_size_t(_s * ctypes.sizeof(ctypes.c_int)), 'cudaMemcpyDeviceToHost')


print gpu_occ.data[86 * COM.layers_per_cell], "layers per cell", COM.layers_per_cell

host_shifts = host.Array(ncomp=26*3, dtype=ctypes.c_double)
cuda_runtime.cuda_mem_cpy(host_shifts.ctypes_data,
                          cuda_halo.HALOS.get_position_shifts.ctypes_data,
                          ctypes.c_size_t(ctypes.sizeof(ctypes.c_double) * 26 * 3),
                          'cudaMemcpyDeviceToHost')
print "SHIFTS", host_shifts.data

print "DAT", sim1.state.d_positions.ctypes_data, sim1.state.d_positions.struct.ptr


_h = '''
     #include <cuda_generic.h>
     extern "C" int test(cuda_Array<double> d_shift);
     '''

_s = '''
     int test(cuda_Array<double> d_shift){
        double h_shift[26*3];
        cudaMemcpy(h_shift, d_shift.ptr, 26*3*sizeof(double), cudaMemcpyDeviceToHost);

        printf("%f \\n", h_shift[0]);


        return 0;
     }
     '''

test_lib = cuda_build.simple_lib_creator(_h,_s,'test')['test']
test_lib(cuda_halo.HALOS.get_position_shifts.struct)

print "AFTER TEST LIB"







cell_contents_count = host.Array(ncomp=COM.cell_contents_count.ncomp, dtype=ctypes.c_int)
cuda_runtime.cuda_mem_cpy(cell_contents_count.ctypes_data, COM.cell_contents_count.ctypes_data, ctypes.c_size_t(cell_contents_count.ncomp * ctypes.sizeof(ctypes.c_int)), 'cudaMemcpyDeviceToHost')
print "CELL contents count BEFORE", cell_contents_count.data

sim1.state.d_positions.halo_exchange()
sim1.state.positions.resize(sim1.state.d_positions.nrow)
cuda_runtime.cuda_mem_cpy(sim1.state.positions.ctypes_data, sim1.state.d_positions.ctypes_data, ctypes.c_size_t(sim1.state.d_positions.nrow * sim1.state.d_positions.ncol * ctypes.sizeof(ctypes.c_double)), 'cudaMemcpyDeviceToHost')

print "DAT after halo exchange", sim1.state.positions.data

cuda_runtime.cuda_mem_cpy(cell_contents_count.ctypes_data, COM.cell_contents_count.ctypes_data, ctypes.c_size_t(cell_contents_count.ncomp * ctypes.sizeof(ctypes.c_int)), 'cudaMemcpyDeviceToHost')
print "CELL contents count AFTER", cell_contents_count.data









