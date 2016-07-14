import numpy as np
import ctypes

from ppmd import *
from ppmd.cuda import *



# n=25 reasonable size
n = 100
N = n**3

#N = 2 # uncomment for 2 bounce


# n=860
rho = 0.2
mu = 1.0
nsig = 5.0

# Initialise basic domain
test_domain = domain.BaseDomainHalo()

# Initialise LJ potential
#test_potential = potential.LennardJones(sigma=1.0,epsilon=1.0)
test_potential = potential.TestPotential1()

test_pos_init = simulation.PosInitLatticeNRhoRand(N,rho,0.,None)

# uncommment for bounce
#test_pos_init = simulation.PosInitTwoParticlesInABox(rx=0.3, extent=np.array([30., 30., 30.]), axis=np.array([1,1,1]))


test_vel_init = simulation.VelInitNormDist(mu,nsig)

# uncomment for bounce
#test_vel_init = simulation.VelInitTwoParticlesInABox(vx=np.array([0., 0., 0.]), vy=np.array([0., 0., 0.]))

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
sim1.state.d_forces = cuda_data.ParticleDat(initial_value=sim1.state.forces, name='forces')
sim1.state.d_velocities = cuda_data.ParticleDat(initial_value=sim1.state.velocities, name='velocities')
sim1.state.d_u = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_double, name='potential_energy')

# print sim1.state.positions.max_npart, sim1.state.positions.npart_local, sim1.state.positions.data

COM.setup(sim1.state.as_func('npart_local'), sim1.state.d_positions, sim1.state.domain)

COM.sort()
cuda_halo.HALOS = cuda_halo.CartesianHalo(COM)



sim1.state.d_positions.halo_exchange()





neighbour_list = cuda_cell.NeighbourListLayerBased(COM, test_potential.rc)
neighbour_list.update()


dat_map = {'P': sim1.state.d_positions(access.R), 'A': sim1.state.d_forces(access.INC0), 'u': sim1.state.d_u(access.INC)}

pair_loop = cuda_pairloop.PairLoopNeighbourList(kernel_in=test_potential.kernel, #_gpu.kernel,
                                                particle_dat_dict=dat_map,
                                                neighbour_list=neighbour_list)
print "n =", sim1.state.d_positions.npart_local
pair_loop.execute(n=sim1.state.d_positions.npart_local)

# Comparisons ---------------------------------------------
sim1.forces_update()
sim1.state.h_forces = data.ParticleDat(npart=sim1.state.d_positions.nrow, ncomp=3, dtype=ctypes.c_double)


cuda_runtime.cuda_mem_cpy(sim1.state.h_forces.ctypes_data, sim1.state.d_forces.ctypes_data, ctypes.c_size_t(sim1.state.d_forces.size), 'cudaMemcpyDeviceToHost')


passed = True

for ix in range(N):
    err = np.linalg.norm(sim1.state.h_forces.data[ix,::] - sim1.state.forces.data[ix,::])

    if err > 10**(-13):
        passed = False
        print err, sim1.state.h_forces.data[ix,::], sim1.state.forces.data[ix,::]

if passed:
    print "Test PASSED"
else:
    print "Test FAILED <------------"

























