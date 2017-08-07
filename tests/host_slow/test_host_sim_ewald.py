#!/usr/bin/python

import ctypes

import numpy as np
import pytest

import ppmd as md

#import matplotlib.pyplot as plt

VERBOSE = True

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
State = md.state.State
ParticleLoop = md.loop.ParticleLoop
Pairloop = md.pairloop.PairLoopNeighbourListNS
PBC = md.domain.BoundaryTypePeriodic()
State = md.state.State


vv_kernel1_code = '''
const double M_tmp = 1.0/M.i[0];
V.i[0] += dht*F.i[0]*M_tmp;
V.i[1] += dht*F.i[1]*M_tmp;
V.i[2] += dht*F.i[2]*M_tmp;
P.i[0] += dt*V.i[0];
P.i[1] += dt*V.i[1];
P.i[2] += dt*V.i[2];
'''

vv_kernel2_code = '''
const double M_tmp = 1.0/M.i[0];
V.i[0] += dht*F.i[0]*M_tmp;
V.i[1] += dht*F.i[1]*M_tmp;
V.i[2] += dht*F.i[2]*M_tmp;
k[0] += (V.i[0]*V.i[0] + V.i[1]*V.i[1] + V.i[2]*V.i[2])*0.5*M.i[0];
'''


@pytest.mark.skip(reason='dev test')
def test_host_sim_2():

    directiona = (1, 0, 0)

    A = State()
    dt = 0.001
    steps = 2000
    shell_steps = 1
    v = 1.0
    A.npart = 2
    E = 50.
    extent = (E, E, E)
    A.domain = md.domain.BaseDomainHalo(extent=extent)
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()

    # init state
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.mass = ParticleDat(ncomp=1)
    A.q = ParticleDat(ncomp=1)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.q[0, 0] = -1.0
    A.q[1, 0] = 1.0

    A.u = ScalarArray(ncomp=1)
    A.ke = ScalarArray(ncomp=1)
    A.crr = ScalarArray(ncomp=1)
    A.cri = ScalarArray(ncomp=1)
    A.crs = ScalarArray(ncomp=1)

    A.gid[:, 0] = np.arange(A.npart)

    A.p[0, :] = E*0.025*np.array(directiona)
    A.p[1, :] = -0.025*E*np.array(directiona)

    A.v[0, :] = np.array(directiona) * v * -1.
    A.v[1, :] = np.array(directiona) * v

    A.f[:] = 0.0

    A.mass[:, 0] = 40.

    A.npart_local = A.npart
    A.filter_on_domain_boundary()

    potaa_rc = 2.
    potaa_rn = potaa_rc * 1.1
    delta = potaa_rn - potaa_rc

    potaa = md.utility.potential.VLennardJones(
        epsilon=1.0,
        sigma=1.0,
        rc=potaa_rc
    )

    potaa_force_updater = md.pairloop.PairLoopNeighbourListNS(
        kernel=potaa.kernel,
        dat_dict=potaa.get_data_map(
            positions=A.p,
            forces=A.f,
            potential_energy=A.u
        ),
        shell_cutoff=potaa_rn
    )

    constants = [
        md.kernel.Constant('dt', dt),
        md.kernel.Constant('dht', 0.5*dt),
    ]

    vv_kernel1 = md.kernel.Kernel('vv1', vv_kernel1_code, constants)
    vv_p1 = ParticleLoop(
        kernel=vv_kernel1,
        dat_dict={'P': A.p(md.access.W),
                  'V': A.v(md.access.W),
                  'F': A.f(md.access.R),
                  'M': A.mass(md.access.R)}
    )

    vv_kernel2 = md.kernel.Kernel('vv2', vv_kernel2_code, constants)
    vv_p2 = ParticleLoop(
        kernel=vv_kernel2,
        dat_dict={'V': A.v(md.access.W),
                  'F': A.f(md.access.R),
                  'M': A.mass(md.access.R),
                  'k': A.ke(md.access.INC0)}
    )

    # ---------------------------------------
    # ewald class instance
    c = md.coulomb.ewald_half.EwaldOrthoganalHalf(
        domain=A.domain,
        eps=10.**-6,
        real_cutoff=10,
        shared_memory=False,
        shell_width=0.5,
        force_unit=10.,
        energy_unit=10.
    )


    # convert to eV
    qfactor = c.internal_to_ev()
    dltfactor = 0.0001957528385665669
    dlevfactor = 1./(96.485*100)
    electro_dl1jmolfactor = 9648.5

    ke_list = []
    u_list = []
    q_list = []
    it_list = []

    for it in md.method.IntegratorRange(
            steps, dt, A.v, shell_steps, delta, verbose=False):

        # velocity verlet 1
        vv_p1.execute(A.npart_local)

        # vdw
        A.f[:,:] = 0
        potaa_force_updater.execute()

        #c.evaluate_contributions(positions=A.p, charges=A.q)
        #A.cri[0] = 0.0
        #c.extract_forces_energy_reciprocal(A.p, A.q, A.f, A.cri)
        A.crr[0] = 0.0
        c.extract_forces_energy_real(A.p, A.q, A.f, A.crr)

        # velocity verlet 2
        vv_p2.execute(A.npart_local)

        if it % 1 == 0:
            it_list.append(it)
            ke_list.append(A.ke[0])
            u_list.append(A.u[0])
            q_list.append(A.crr[0] + A.cri[0])
            print(A.crr[0] + A.cri[0])


    it_array = md.mpi.all_reduce(np.array(it_list))
    ke_array = md.mpi.all_reduce(np.array(ke_list))
    u_array = md.mpi.all_reduce(np.array(u_list))
    q_array = md.mpi.all_reduce(np.array(q_list))

    plt.plot(it_array, ke_array, color='r')
    plt.plot(it_array, u_array, color='g')
    plt.plot(it_array, q_array, color='b')
    plt.plot(it_array, q_array+ke_array+u_array, color='k')
    plt.show()

    A.gather_data_on(0)
    if rank == 0:
        for ix in range(len(ke_array)-1):
            ke_diff = ke_array[ix] - ke_array[ix+1]
            u_diff = u_array[ix] - u_array[ix+1]
            assert abs(ke_diff) < 4.
            assert abs(u_diff) < 4.

            steperr = abs(ke_diff + u_diff)
            #if VERBOSE and steperr > 10.**-5:
            #    print steperr
            assert abs(ke_diff + u_diff) < 10.**-3

        assert abs(ke_array[0] + u_array[0] - ke_array[-1] - u_array[-1]) < 2.*10.**-3
























