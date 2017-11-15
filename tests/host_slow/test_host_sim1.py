#!/usr/bin/python

import ctypes

import numpy as np
import os
import pytest

import ppmd as md

N = 1000
crN = 10 #cubert(N)
E = 8.

Eo2 = E/2.

tol = 10.**(-14)

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
GlobalArray = md.data.GlobalArray
State = md.state.State

@pytest.fixture
def state():
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)
    A.nc = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A


@pytest.fixture(scope="module", params=list({0, nproc-1}))
def base_rank(request):
    return request.param

@pytest.fixture(scope="module", params=list({
    md.pairloop.PairLoopNeighbourListNS,
    md.pairloop.CellByCellOMP}))
def PAIRLOOP(request):
    return request.param


@pytest.mark.slowtest
def test_host_sim_1():
    PAIRLOOP = md.pairloop.PairLoopNeighbourListNS

    A = State()


    DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../res/1k_1k_08442_25/'
    )

    CONFIG = DIR + '/CONFIG'
    CONTROL = DIR + '/CONTROL'
    FIELD = DIR + '/FIELD'

    rFIELD = md.utility.dl_poly.read_field(FIELD)
    rCONTROL = md.utility.dl_poly.read_control(CONTROL)

    A.npart = int(md.utility.dl_poly.get_field_value(rFIELD, 'NUMMOLS')[0][0])


    extent = md.utility.dl_poly.read_domain_extent(CONFIG)
    A.domain = md.domain.BaseDomainHalo(extent=extent)
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()


    # init state
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.mass = ParticleDat(ncomp=1)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)


    A.gid[:,0] = np.arange(A.npart)
    A.p[:] = md.utility.dl_poly.read_positions(CONFIG)
    A.v[:] = md.utility.dl_poly.read_velocities(CONFIG)
    A.f[:] = md.utility.dl_poly.read_forces(CONFIG)
    A.u[0] = 0.0

    A.mass[:,0] = np.ones(A.npart) * \
                float(md.utility.dl_poly.get_field_value(
                    rFIELD,
                    'Ar'
                )[0][0])

    A.npart_local = A.npart
    A.filter_on_domain_boundary()

    potaa_rc = float(md.utility.dl_poly.get_control_value(rCONTROL, 'cutoff')[0][0])
    potaa_rn = potaa_rc * 1.1
    potaa = md.utility.potential.Buckingham(
        a=1.69*10**-8.0,
        b=1.0/0.273,
        c=102*10**-12,
        rc=potaa_rc
    )

    potaa_force_updater = PAIRLOOP(
        kernel = potaa.kernel,
        dat_dict= potaa.get_data_map(
            positions=A.p,
            forces=A.f,
            potential_energy=A.u
        ),
        shell_cutoff=potaa_rn
    )

    potaa_kinetic_energy_updater = md.utility.high_method.KineticEnergyTracker(
        velocities=A.v,
        masses=A.mass
    )

    potaa_potential_energy = md.utility.high_method.PotentialEnergyTracker(
        potential_energy_dat=A.u
    )

    potaa_schedule = md.utility.high_method.Schedule(
        [1, 1],
        [potaa_kinetic_energy_updater.execute, potaa_potential_energy.execute]
    )

    potaa_integrator = md.utility.high_method.IntegratorVelocityVerlet(
        positions=A.p,
        forces=A.f,
        velocities=A.v,
        masses=A.mass,
        force_updater=potaa_force_updater,
        interaction_cutoff=potaa_rc,
        list_reuse_count=10
    )

    potaa_integrator.integrate(0.0001, .1, potaa_schedule)

    eng_err = abs(potaa_kinetic_energy_updater.get_kinetic_energy_array()[0] + \
        potaa_potential_energy.get_potential_energy_array()[0] - \
        potaa_kinetic_energy_updater.get_kinetic_energy_array()[-1] - \
        potaa_potential_energy.get_potential_energy_array()[-1])

    print("err:", eng_err)



@pytest.mark.slowtest
def test_host_sim_2():
    PAIRLOOP = md.pairloop.CellByCellOMP

    A = State()

    DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../res/1k_1k_08442_25/'
    )

    CONFIG = DIR + '/CONFIG'
    CONTROL = DIR + '/CONTROL'
    FIELD = DIR + '/FIELD'

    rFIELD = md.utility.dl_poly.read_field(FIELD)
    rCONTROL = md.utility.dl_poly.read_control(CONTROL)

    A.npart = int(md.utility.dl_poly.get_field_value(rFIELD, 'NUMMOLS')[0][0])


    extent = md.utility.dl_poly.read_domain_extent(CONFIG)
    A.domain = md.domain.BaseDomainHalo(extent=extent)
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()


    # init state
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.mass = ParticleDat(ncomp=1)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = GlobalArray(ncomp=1)

    A.gid[:,0] = np.arange(A.npart)
    A.p[:] = md.utility.dl_poly.read_positions(CONFIG)
    A.v[:] = md.utility.dl_poly.read_velocities(CONFIG)
    A.f[:] = md.utility.dl_poly.read_forces(CONFIG)

    A.mass[:,0] = np.ones(A.npart) * \
                float(md.utility.dl_poly.get_field_value(
                    rFIELD,
                    'Ar'
                )[0][0])

    A.npart_local = A.npart
    A.filter_on_domain_boundary()

    potaa_rc = float(md.utility.dl_poly.get_control_value(
        rCONTROL, 'cutoff')[0][0])
    potaa_rn = potaa_rc * 1.1
    potaa = md.utility.potential.Buckingham(
        a=1.69*10**-8.0,
        b=1.0/0.273,
        c=102*10**-12,
        rc=potaa_rc
    )

    potaa_force_updater = PAIRLOOP(
        kernel = potaa.kernel,
        dat_dict= potaa.get_data_map(
            positions=A.p,
            forces=A.f,
            potential_energy=A.u
        ),
        shell_cutoff=potaa_rn
    )

    potaa_kinetic_energy_updater = md.utility.high_method.KineticEnergyTracker(
        velocities=A.v,
        masses=A.mass
    )

    potaa_potential_energy = md.utility.high_method.PotentialEnergyTracker(
        potential_energy_dat=A.u
    )

    potaa_schedule = md.utility.high_method.Schedule(
        [1, 1],
        [potaa_kinetic_energy_updater.execute, potaa_potential_energy.execute]
    )

    potaa_integrator = md.utility.high_method.IntegratorVelocityVerlet(
        positions=A.p,
        forces=A.f,
        velocities=A.v,
        masses=A.mass,
        force_updater=potaa_force_updater,
        interaction_cutoff=potaa_rc,
        list_reuse_count=10
    )

    potaa_integrator.integrate(0.0001, .1, potaa_schedule)

    eng_err = abs(potaa_kinetic_energy_updater.get_kinetic_energy_array()[0] + \
        potaa_potential_energy.get_potential_energy_array()[0] - \
        potaa_kinetic_energy_updater.get_kinetic_energy_array()[-1] - \
        potaa_potential_energy.get_potential_energy_array()[-1])

    print("err:", eng_err)




