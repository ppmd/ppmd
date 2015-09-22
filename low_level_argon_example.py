#!/usr/bin/python
from ppmd import *

# choose a domain.
periodic_domain = domain.BaseDomainHalo()


# N: Number of particles.
n = 10
N = n ** 3

# position initialisation method
position_init = simulation.PosInitLatticeNRhoRand(N, rho=0.2, dev=0.)

# velocity initialisation method
velocity_init = simulation.VelInitNormDist(mu=0., sig=5.)

# mass initialisation method
mass_init = simulation.MassInitIdentical(m=39.948)


# Combine the existing intialisations into a simulation. We do not pass a potential ths sets up a cell structure using the passed cutoff.

sim = simulation.BaseMDSimulation(domain_in=periodic_domain,
                                  particle_pos_init=position_init,
                                  particle_vel_init=velocity_init,
                                  particle_mass_init=mass_init,
                                  n=N,
                                  cutoff=8.5
                                  )


# create a custom potential using pairlooping.

kernel_code = '''

const double R0 = P[1][0] - P[0][0];
const double R1 = P[1][1] - P[0][1];
const double R2 = P[1][2] - P[0][2];

const double r2 = R0*R0 + R1*R1 + R2*R2;

if (r2 < rc2){

    const double r_m2 = sigma2/r2;
    const double r_m4 = r_m2*r_m2;
    const double r_m6 = r_m4*r_m2;
    
    u[0]+= CV*((r_m6-1.0)*r_m6 + 0.25);
    
    const double r_m8 = r_m4*r_m4;
    const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

    
    A[0][0]+=f_tmp*R0;
    A[0][1]+=f_tmp*R1;
    A[0][2]+=f_tmp*R2;
    
    A[1][0]-=f_tmp*R0;
    A[1][1]-=f_tmp*R1;
    A[1][2]-=f_tmp*R2;

}

'''

# setup kernel constants
sigma = 3.405
epsilon = 0.9661
cutoff = 8.5

kernel_constants = (kernel.Constant('sigma2', sigma ** 2),
                    kernel.Constant('rc2', cutoff ** 2),
                    kernel.Constant('CF', -48 * epsilon / sigma ** 2),
                    kernel.Constant('CV', 4. * epsilon))

LJ_kernel = kernel.Kernel('custom_lennard_jones', 
                          kernel_code, 
                          kernel_constants)

kernel_dat_dict = {'P': sim.state.positions(access.R), 
                   'A': sim.state.forces(access.W), 
                   'u': sim.state.u(access.INC)}


force_update_pairloop = pairloop.PairLoopRapaportHalo(domain=periodic_domain,
                                                      kernel=LJ_kernel,
                                                      dat_dict=kernel_dat_dict)

# In future, access descriptors should avoid this being called by the user.
cell.cell_list.sort()

# update forces and potential energy.
force_update_pairloop.execute()




# create a verlocity verlet integrator.

dt = 0.0001

vv_kernel1_code = '''
const double M_tmp = 1/M[0];
V[0] += dht*A[0]*M_tmp;
V[1] += dht*A[1]*M_tmp;
V[2] += dht*A[2]*M_tmp;
P[0] += dt*V[0];
P[1] += dt*V[1];
P[2] += dt*V[2];
'''
        
vv_kernel2_code = '''
const double M_tmp = 1/M[0];
V[0] += dht*A[0]*M_tmp;
V[1] += dht*A[1]*M_tmp;
V[2] += dht*A[2]*M_tmp;
'''
vv_constants = (kernel.Constant('dt', dt), 
                kernel.Constant('dht',0.5 * dt))
                
vv_dat_dict = {'P':sim.state.positions(access.RW),
               'V':sim.state.velocities(access.RW),
               'A':sim.state.forces(access.R),
               'M':sim.state.mass(access.R)}

vv_kernel1 = kernel.Kernel('vv1', vv_kernel1_code, vv_constants)
vv_kernel2 = kernel.Kernel('vv2', vv_kernel2_code, vv_constants)

vv_part1_loop = loop.SingleAllParticleLoop(sim.state.as_func('n'), 
                                           sim.state.types,
                                           vv_kernel1,
                                           vv_dat_dict)

vv_part2_loop = loop.SingleAllParticleLoop(sim.state.as_func('n'),
                                           sim.state.types,
                                           vv_kernel2,
                                           vv_dat_dict)




# Integrate system forward in time.

for ix in range(100):

    # first part of velocity verlet.
    vv_part1_loop.execute()
    
    # To be handled by access descriptiors in future.
    periodic_domain.bc_execute()
    
    
    # update forces using custom potential.
    
    # To be handled by access descriptiors in future.
    cell.cell_list.sort()
    sim.state.positions.halo_exchange()
    
    
    # update forces and potential energy using custom pairloop.
    force_update_pairloop.execute()

    
    # second part of velocity verlet.
    vv_part2_loop.execute()
















