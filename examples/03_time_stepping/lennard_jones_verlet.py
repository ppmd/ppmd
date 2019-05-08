# Lennard-Jones example to compute forces and system potential energy


import numpy as np
from ctypes import *
from ppmd import *
import sys
import time

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
IntegratorRange = method.IntegratorRange



def make_force_pairloop(A, lj_energy, constants, cutoff):

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
        shell_cutoff=cutoff
    )

    return lj_pairloop


def make_velocity_verlet_loop_1(A, constants):

    vv_kernel1_code = '''
    const double M_tmp = 1.0/M.i[0];
    V.i[0] += dht*F.i[0]*M_tmp;
    V.i[1] += dht*F.i[1]*M_tmp;
    V.i[2] += dht*F.i[2]*M_tmp;
    P.i[0] += dt*V.i[0];
    P.i[1] += dt*V.i[1];
    P.i[2] += dt*V.i[2];
    '''
    vv_kernel1 = Kernel('vv1', vv_kernel1_code, constants)
    vv_p1 = ParticleLoop(
        kernel=vv_kernel1,
        dat_dict={'P': A.pos(access.INC),
                  'V': A.vel(access.INC),
                  'F': A.force(access.READ),
                  'M': A.mass(access.READ)}
    )

    return vv_p1


def make_velocity_verlet_loop_2(A, ke_energy, constants):

    vv_kernel2_code = '''
    const double M_tmp = 1.0/M.i[0];
    V.i[0] += dht*F.i[0]*M_tmp;
    V.i[1] += dht*F.i[1]*M_tmp;
    V.i[2] += dht*F.i[2]*M_tmp;
    k[0] += (V.i[0]*V.i[0] + V.i[1]*V.i[1] + V.i[2]*V.i[2])*0.5*M.i[0];
    '''
    vv_kernel2 = Kernel('vv2', vv_kernel2_code, constants)
    vv_p2 = ParticleLoop(
        kernel=vv_kernel2,
        dat_dict={'V': A.vel(access.INC),
                  'F': A.force(access.READ),
                  'M': A.mass(access.READ),
                  'k': ke_energy(access.INC_ZERO)}
    )

    return vv_p2



if __name__ == '__main__':


    # Some parameters
    steps = 10000
    dt = 0.001
    shell_steps = 10
    N1 = 10
    
    if len(sys.argv) > 1:
        N1 = int(sys.argv[1])
        steps = int(sys.argv[2])

    N = N1**3
    
    


    # PPMD must be able to decompose the domain into cells with at least 3 cells per
    # dimension
    r_c = 6.
    E = N1 * r_c * 0.75
    E = max(E, r_c * 4)


    # Lennard Jones parameters
    epsilon = 1.0
    sigma = 1.0

    # in MD simulations it is common to use neighbour lists or cell decomposions
    # built with a cutoff larger than the interaction cutoff.
    r_n = r_c + 0.1*r_c
    delta = r_n - r_c

    # kernel constants
    constants = (
        Constant('CV', 4. * epsilon),
        Constant('CF', -48 * epsilon / sigma ** 2),
        Constant('sigma2', sigma ** 2),
        Constant('rc2', r_c ** 2),
        Constant('internalshift', (sigma / r_c) ** 6.0 - (sigma / r_c) ** 12.0),
        Constant('dt', dt),
        Constant('dht', 0.5*dt),
    )

    # make a state object and set the global number of particles N
    A = State()

    # give the state a domain and boundary condition
    A.domain = domain.BaseDomainHalo(extent=(E, E, E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    # add a PositionDat to contain positions
    A.pos = PositionDat(ncomp=3)
    A.vel = ParticleDat(ncomp=3)
    A.mass = ParticleDat(ncomp=1)
    A.force = ParticleDat(ncomp=3)

    # system energy store
    lj_energy = GlobalArray(ncomp=1, dtype=c_double)
    ke_energy = GlobalArray(ncomp=1, dtype=c_double)

    
    # on MPI rank 0 add a cubic lattice of particles to the system 
    # with standard normal velocities
    rng = np.random.RandomState(512)
    with A.modify() as AM:
        if A.domain.comm.rank == 0:
            AM.add({
                A.pos: utility.lattice.cubic_lattice((N1, N1, N1), (E, E, E)),
                A.vel: rng.normal(0, 1.0, size=(N, 3)),
                A.mass: np.ones(shape=(N, 1))
            })


    lj_pairloop = make_force_pairloop(A, lj_energy, constants, r_n)
    vv1 = make_velocity_verlet_loop_1(A, constants)
    vv2 = make_velocity_verlet_loop_2(A, ke_energy, constants)


    # main timestepping loop
    t0 = time.time()
    for it in IntegratorRange(steps, dt, A.vel, shell_steps, delta, verbose=False):
        
        vv1.execute()
        lj_pairloop.execute()
        vv2.execute()
        

        if A.domain.comm.rank == 0 and it % 10 == 0:
            print("{: 8d} | {: 12.8e} {: 12.8e} | {: 12.8e}".format(it, lj_energy[0], ke_energy[0], lj_energy[0] + ke_energy[0]))
    t1 = time.time()

    if A.domain.comm.rank == 0:
        print("Time taken: \t", t1 - t0)










