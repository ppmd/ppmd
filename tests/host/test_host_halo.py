#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md

N = 1000
crN = 10 #cubert(N)
E = 8.

Eo2 = E/2.
tol = 0.1


rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()


PositionDat = md.data.PositionDat
ParticleDat = md.data.ParticleDat
ScalarArray = md.data.ScalarArray
State = md.state.State





@pytest.fixture
def s_nd():
    """
    State with no domain, hence will not spatially decompose
    """

    A = State()
    A.npart = N
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A


@pytest.fixture(scope="module", params=(0, nproc-1))
def base_rank(request):
    return request.param

@pytest.fixture
def state_int_dat():
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.int = ParticleDat(ncomp=2, dtype=ctypes.c_long)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

@pytest.fixture
def state():
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = md.domain.BoundaryTypePeriodic()
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

def test_host_halo_cube_1(state):
    """
    Check cell counts before and after halo exchange.
    """
    cell_width = float(E)/float(crN)
    state.domain.cell_decompose(cell_width)

    state.get_cell_to_particle_map().create()
    state.get_cell_to_particle_map().update_required = True


    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1


    state.p[:] = pi
    state.npart_local = N

    state.filter_on_domain_boundary()

    state.get_cell_to_particle_map().check()

    ca = state.domain.cell_array



    for cx in range(ca[0]):
        for cy in range(ca[1]):
            for cz in range(ca[2]):

                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                if ( cx == 0 or cx == ca[0]-1 ) or \
                   ( cy == 0 or cy == ca[1]-1 ) or \
                   ( cz == 0 or cz == ca[2]-1 ):
                    assert state.get_cell_to_particle_map().cell_contents_count[ci] == 0
                else:
                    assert state.get_cell_to_particle_map().cell_contents_count[ci] == 1

    state.p.halo_exchange()


    for cx in range(ca[0]):
        for cy in range(ca[1]):
            for cz in range(ca[2]):
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                assert state.get_cell_to_particle_map().cell_contents_count[ci] == 1





def test_host_halo_cube_2(state):
    """
    Check cell contents of a simple cube by value.
    """
    cell_width = float(E)/float(crN)
    state.domain.cell_decompose(cell_width)
    state.get_cell_to_particle_map().create()
    state.get_cell_to_particle_map().update_required = True


    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()

    state.get_cell_to_particle_map().check()


    state.p.halo_exchange()

    pj = state.p[:state.p.npart_local+state.p.npart_local_halo:]
    pj = pj[np.lexsort((pj[:, 0], pj[:, 1], pj[:,2]))]

    ca = state.domain.cell_array



    offsets = (-1, 0, 1)
    disp = float(E)/float(crN)

    cax = (1, ca[0]-2)
    cay = (1, ca[0]-2)
    caz = (1, ca[0]-2)

    for cx in cax:
        for cy in cay:
            for cz in caz:
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                rci = pj[ci]
                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:

                            cj = (cz+oz)*(ca[0]*ca[1]) + (cy+oy)*ca[0] + (cx+ox)
                            cval = np.array((ox*disp, oy*disp, oz*disp)) + rci

                            #print rank, rci, cval, (ox,oy,oz), cj, pj[cj], pj[cj] - cval
                            assert np.max(np.abs(pj[cj] - cval)) < 2.*(10.**(-15.))




def test_host_halo_cube_3(state):
    """
    Check cell contents using a cell by cell inspection.
    """
    cell_width = float(E)/float(crN)
    state.domain.cell_decompose(cell_width)
    state.get_cell_to_particle_map().create()
    state.get_cell_to_particle_map().update_required = True


    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.npart_local = N
    state.filter_on_domain_boundary()

    state.get_cell_to_particle_map().check()


    state.p.halo_exchange()

    pj = state.p[:state.p.npart_local+state.p.npart_local_halo:]

    ca = state.domain.cell_array


    offsets = (-1, 0, 1)
    disp = float(E)/float(crN)

    cax = (1, ca[0]-2)
    cay = (1, ca[0]-2)
    caz = (1, ca[0]-2)

    cl = state.get_cell_to_particle_map().cell_list
    end = state.get_cell_to_particle_map().offset.value

    for cx in cax:
        for cy in cay:
            for cz in caz:
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                px = cl[end+ci]
                rci = pj[px]

                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:

                            cj = (cz+oz)*(ca[0]*ca[1]) + (cy+oy)*ca[0] + (cx+ox)

                            py = cl[end+cj]
                            rcj = pj[py]

                            cval = np.array((ox*disp, oy*disp, oz*disp)) + rci

                            assert np.max(np.abs(rcj - cval)) < 2.*(10.**(-15.))



def test_host_halo_cube_non_position_1(state):
    """
    Check cell contents using a cell by cell inspection.
    """
    cell_width = float(E)/float(crN)
    state.domain.cell_decompose(cell_width)
    state.get_cell_to_particle_map().create()
    state.get_cell_to_particle_map().update_required = True


    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state.p[:] = pi
    state.v[:] = pi

    state.npart_local = N
    state.filter_on_domain_boundary()

    state.get_cell_to_particle_map().check()

    state.p.halo_exchange()
    state.v.halo_exchange()

    pj = state.p[:state.p.npart_local+state.p.npart_local_halo:]
    vj = state.v[:state.v.npart_local+state.v.npart_local_halo:]

    ca = state.domain.cell_array


    offsets = (-1, 0, 1)
    disp = float(E)/float(crN)

    cax = (1, ca[0]-2)
    cay = (1, ca[0]-2)
    caz = (1, ca[0]-2)

    # check the v halo exchange is as expected
    cl = state.get_cell_to_particle_map().cell_list
    end = state.get_cell_to_particle_map().offset.value
    for cx in cax:
        for cy in cay:
            for cz in caz:
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                px = cl[end+ci]
                rci = vj[px]

                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:

                            cj = (cz+oz)*(ca[0]*ca[1]) + (cy+oy)*ca[0] + (cx+ox)

                            py = cl[end+cj]
                            rcj = vj[py]

                            cval = np.array((ox*disp, oy*disp, oz*disp)) + rci

                            # because the v contains unshifted positions we
                            # apply a correction if the particle is outside the
                            # domain

                            if cval[0] < -0.5*E:
                                cval[0] += E
                            elif cval[0] > 0.5*E:
                                cval[0] -= E
                            if cval[1] < -0.5*E:
                                cval[1] += E
                            elif cval[1] > 0.5*E:
                                cval[1] -= E
                            if cval[2] < -0.5*E:
                                cval[2] += E
                            elif cval[2] > 0.5*E:
                                cval[2] -= E

                            assert np.max(np.abs(rcj - cval)) < 2.*(10.**(-15.))


    # check the p halo exchange is as expected
    for cx in cax:
        for cy in cay:
            for cz in caz:
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                px = cl[end+ci]
                rci = pj[px]

                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:

                            cj = (cz+oz)*(ca[0]*ca[1]) + (cy+oy)*ca[0] + (cx+ox)

                            py = cl[end+cj]
                            rcj = pj[py]

                            cval = np.array((ox*disp, oy*disp, oz*disp)) + rci

                            assert np.max(np.abs(rcj - cval)) < 2.*(10.**(-15.))



def test_host_halo_cube_non_position_2(state_int_dat):
    """
    Check cell contents using a cell by cell inspection.
    """
    cell_width = float(E)/float(crN)
    state_int_dat.domain.cell_decompose(cell_width)
    state_int_dat.get_cell_to_particle_map().create()
    state_int_dat.get_cell_to_particle_map().update_required = True


    pi = np.zeros([N,3], dtype=ctypes.c_double)
    px = 0

    # This is upsetting....
    for ix in range(crN):
        for iy in range(crN):
            for iz in range(crN):
                pi[px,:] = (E/crN)*np.array([ix, iy, iz]) - 0.5*(E-E/crN)*np.ones(3)
                px += 1

    state_int_dat.p[:] = pi
    state_int_dat.v[:] = pi

    state_int_dat.npart_local = N
    state_int_dat.filter_on_domain_boundary()

    state_int_dat.get_cell_to_particle_map().check()

    state_int_dat.p.halo_exchange()
    state_int_dat.v.halo_exchange()

    pj = state_int_dat.p[:state_int_dat.p.npart_local+state_int_dat.p.npart_local_halo:]
    vj = state_int_dat.v[:state_int_dat.v.npart_local+state_int_dat.v.npart_local_halo:]

    ca = state_int_dat.domain.cell_array


    offsets = (-1, 0, 1)
    disp = float(E)/float(crN)

    cax = (1, ca[0]-2)
    cay = (1, ca[0]-2)
    caz = (1, ca[0]-2)

    # check the v halo exchange is as expected
    cl = state_int_dat.get_cell_to_particle_map().cell_list
    end = state_int_dat.get_cell_to_particle_map().offset.value
    for cx in cax:
        for cy in cay:
            for cz in caz:
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                px = cl[end+ci]
                rci = vj[px]

                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:

                            cj = (cz+oz)*(ca[0]*ca[1]) + (cy+oy)*ca[0] + (cx+ox)

                            py = cl[end+cj]
                            rcj = vj[py]

                            cval = np.array((ox*disp, oy*disp, oz*disp)) + rci

                            # because the v contains unshifted positions we
                            # apply a correction if the particle is outside the
                            # domain

                            if cval[0] < -0.5*E:
                                cval[0] += E
                            elif cval[0] > 0.5*E:
                                cval[0] -= E
                            if cval[1] < -0.5*E:
                                cval[1] += E
                            elif cval[1] > 0.5*E:
                                cval[1] -= E
                            if cval[2] < -0.5*E:
                                cval[2] += E
                            elif cval[2] > 0.5*E:
                                cval[2] -= E

                            assert np.max(np.abs(rcj - cval)) < 2.*(10.**(-15.))


    # check the p halo exchange is as expected
    for cx in cax:
        for cy in cay:
            for cz in caz:
                ci = cz*(ca[0]*ca[1]) + cy*ca[0] + cx
                px = cl[end+ci]
                rci = pj[px]

                for ox in offsets:
                    for oy in offsets:
                        for oz in offsets:

                            cj = (cz+oz)*(ca[0]*ca[1]) + (cy+oy)*ca[0] + (cx+ox)

                            py = cl[end+cj]
                            rcj = pj[py]

                            cval = np.array((ox*disp, oy*disp, oz*disp)) + rci

                            assert np.max(np.abs(rcj - cval)) < 2.*(10.**(-15.))




