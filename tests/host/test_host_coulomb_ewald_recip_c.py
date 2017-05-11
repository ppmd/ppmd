__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"


import numpy as np
import pytest
from decimal import Decimal
import ppmd as md

import scipy
import scipy.constants
from math import pi, exp

from math import sqrt

mpi_rank = md.mpi.MPI.COMM_WORLD.Get_rank()
mpi_size = md.mpi.MPI.COMM_WORLD.Get_size()
ParticleDat = md.data.ParticleDat

def assert_tol(val, tol, msg="tolerance not met"):
    assert abs(val) < 10.**(-1*tol), msg


def test_ewald_energy_python_co2_1():
    """
    Test that the python implementation of ewald calculates the correct 
    real space contribution and self interaction contribution.
    """

    if mpi_rank > 0:
        return

    eta = 0.26506
    alpha = eta**2.
    rc = 12.

    e = 24.47507
    domain = md.domain.BaseDomainHalo(extent=(e,e,e))
    c = md.coulomb.CoulombicEnergy(domain=domain, real_cutoff=rc, alpha=alpha)

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"

    data = np.load('../res/coulomb/CO2.npy')

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    #positions[:, 0] -= e*0.5
    #positions[:, 1] -= e*0.5
    #positions[:, 2] -= e*0.5
    #print(np.max(positions[:,0]), np.min(positions[:,0]))
    #print(np.max(positions[:,1]), np.min(positions[:,1]))
    #print(np.max(positions[:,2]), np.min(positions[:,2]))

    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.evaluate_lr(positions=positions, charges=charges)
    #print rs
    #assert abs(rs[0]*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "Energy from loop back over particles"
    #assert abs(rs[1]*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "Energy from structure factor"

    py_recip_space = np.load('../res/coulomb/co2_recip_space.npy')


    nmax_x = c._vars['nmax_vec'][0]
    nmax_y = c._vars['nmax_vec'][1]
    nmax_z = c._vars['nmax_vec'][2]
    recip_axis_len = c._vars['recip_axis_len'].value
    recip_vec = c._vars['recip_vec']
    nmax_vec = c._vars['nmax_vec']
    coeff_space = c._vars['coeff_space']
    max_recip = c._vars['max_recip'].value
    alpha = c._vars['alpha'].value
    ivolume = c._vars['ivolume']
    recip_space = c._vars['recip_space_kernel']
    nkmax = c._vars['recip_axis_len'].value
    nkaxis = nkmax


    axes_size = 12*nkaxis
    axes = recip_space[0:axes_size:].view()
    plane_size = 4*nmax_x*nmax_y + 4*nmax_y*nmax_z + 4*nmax_z*nmax_x
    planes = recip_space[axes_size:axes_size+plane_size*2:].view()
    quad_size = nmax_x*nmax_y*nmax_z
    quad_start = axes_size+plane_size*2
    quads = recip_space[quad_start:quad_start+quad_size*16].view()


    #+ve X
    rax = 0
    iax = 6
    rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]
    itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]
    for ix in range(nmax_x):
        assert_tol(rtmp[ix] - py_recip_space[0, nmax_x+ix+1, nmax_y, nmax_z], 8)
        assert_tol(itmp[ix] - py_recip_space[1, nmax_x+ix+1, nmax_y, nmax_z], 8)
    # -ve X
    rax = 2
    iax = 8
    rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]
    itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]
    for ix in range(nmax_x):
        assert_tol(rtmp[ix] - py_recip_space[0, nmax_x-ix-1, nmax_y, nmax_z], 8)
        assert_tol(itmp[ix] - py_recip_space[1, nmax_x-ix-1, nmax_y, nmax_z], 8)
    #+ve y
    rax = 1
    iax = 7
    rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]
    itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]
    for ix in range(nmax_y):
        assert_tol(rtmp[ix] - py_recip_space[0, nmax_x, nmax_y+ix+1, nmax_z], 8)
        assert_tol(itmp[ix] - py_recip_space[1, nmax_x, nmax_y+ix+1, nmax_z], 8)
    #-ve y
    rax = 3
    iax = 9
    rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]
    itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]
    for ix in range(nmax_y):
        assert_tol(rtmp[ix] - py_recip_space[0, nmax_x, nmax_y-ix-1, nmax_z], 8)
        assert_tol(itmp[ix] - py_recip_space[1, nmax_x, nmax_y-ix-1, nmax_z], 8)
    #+ve z
    rax = 4
    iax = 10
    rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]
    itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]
    for ix in range(nmax_z):
        assert_tol(rtmp[ix] - py_recip_space[0, nmax_x, nmax_y, nmax_z+ix+1], 8)
        assert_tol(itmp[ix] - py_recip_space[1, nmax_x, nmax_y, nmax_z+ix+1], 8)
    #-ve z
    rax = 5
    iax = 11
    rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]
    itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]
    for ix in range(nmax_z):
        assert_tol(rtmp[ix] - py_recip_space[0, nmax_x, nmax_y, nmax_z-ix-1], 8)
        assert_tol(itmp[ix] - py_recip_space[1, nmax_x, nmax_y, nmax_z-ix-1], 8)

    # PLANES ------------------------------------------------------------------

    # XY
    tps = nmax_x*nmax_y*4
    rplane = 0
    iplane = tps
    rtmp = planes[rplane:rplane+tps:]
    itmp = planes[iplane:iplane+tps:]

    tmp_rtmp = rtmp[::4]
    tmp_itmp = itmp[::4]
    for ix in range(nmax_x):
        for iy in range(nmax_y):
            assert_tol(tmp_rtmp[iy*nmax_x + ix] - py_recip_space[0, nmax_x+ix+1, nmax_y+iy+1, nmax_z ], 8)
            assert_tol(tmp_itmp[iy*nmax_x + ix] - py_recip_space[1, nmax_x+ix+1, nmax_y+iy+1, nmax_z ], 8)
    tmp_rtmp = rtmp[1::4]
    tmp_itmp = itmp[1::4]
    for ix in range(nmax_x):
        for iy in range(nmax_y):
            assert_tol(tmp_rtmp[iy*nmax_x + ix] - py_recip_space[0, nmax_x-ix-1, nmax_y+iy+1, nmax_z ], 8)
            assert_tol(tmp_itmp[iy*nmax_x + ix] - py_recip_space[1, nmax_x-ix-1, nmax_y+iy+1, nmax_z ], 8)
    tmp_rtmp = rtmp[2::4]
    tmp_itmp = itmp[2::4]
    for ix in range(nmax_x):
        for iy in range(nmax_y):
            assert_tol(tmp_rtmp[iy*nmax_x + ix] - py_recip_space[0, nmax_x-ix-1, nmax_y-iy-1, nmax_z ], 8)
            assert_tol(tmp_itmp[iy*nmax_x + ix] - py_recip_space[1, nmax_x-ix-1, nmax_y-iy-1, nmax_z ], 8)
    tmp_rtmp = rtmp[3::4]
    tmp_itmp = itmp[3::4]
    for ix in range(nmax_x):
        for iy in range(nmax_y):
            assert_tol(tmp_rtmp[iy*nmax_x + ix] - py_recip_space[0, nmax_x+ix+1, nmax_y-iy-1, nmax_z ], 8)
            assert_tol(tmp_itmp[iy*nmax_x + ix] - py_recip_space[1, nmax_x+ix+1, nmax_y-iy-1, nmax_z ], 8)


    # YZ
    rplane = iplane + tps
    tps = nmax_y*nmax_z*4
    iplane = rplane + tps
    rtmp = planes[rplane:rplane+tps:]
    itmp = planes[iplane:iplane+tps:]


    tmp_rtmp = rtmp[::4]
    tmp_itmp = itmp[::4]
    for ix in range(nmax_y):
        for iy in range(nmax_z):
            assert_tol(tmp_rtmp[iy*nmax_y + ix] - py_recip_space[0, nmax_x, nmax_y+ix+1, nmax_z+iy+1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_y + ix] - py_recip_space[1, nmax_x, nmax_y+ix+1, nmax_z+iy+1 ], 8)
    tmp_rtmp = rtmp[1::4]
    tmp_itmp = itmp[1::4]
    for ix in range(nmax_y):
        for iy in range(nmax_z):
            assert_tol(tmp_rtmp[iy*nmax_y + ix] - py_recip_space[0, nmax_x, nmax_y-ix-1, nmax_z+iy+1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_y + ix] - py_recip_space[1, nmax_x, nmax_y-ix-1, nmax_z+iy+1 ], 8)
    tmp_rtmp = rtmp[2::4]
    tmp_itmp = itmp[2::4]
    for ix in range(nmax_y):
        for iy in range(nmax_z):
            assert_tol(tmp_rtmp[iy*nmax_y + ix] - py_recip_space[0, nmax_x, nmax_y-ix-1, nmax_z-iy-1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_y + ix] - py_recip_space[1, nmax_x, nmax_y-ix-1, nmax_z-iy-1 ], 8)
    tmp_rtmp = rtmp[3::4]
    tmp_itmp = itmp[3::4]
    for ix in range(nmax_y):
        for iy in range(nmax_z):
            assert_tol(tmp_rtmp[iy*nmax_y + ix] - py_recip_space[0, nmax_x, nmax_y+ix+1, nmax_z-iy-1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_y + ix] - py_recip_space[1, nmax_x, nmax_y+ix+1, nmax_z-iy-1 ], 8)


    # ZX
    rplane = iplane + tps
    tps = nmax_x*nmax_x*4
    iplane = rplane + tps
    rtmp = planes[rplane:rplane+tps:]
    itmp = planes[iplane:iplane+tps:]

    tmp_rtmp = rtmp[::4]
    tmp_itmp = itmp[::4]
    for ix in range(nmax_z):
        for iy in range(nmax_x):
            assert_tol(tmp_rtmp[iy*nmax_z + ix] - py_recip_space[0, nmax_x+1+iy, nmax_y, nmax_z+ix+1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_z + ix] - py_recip_space[1, nmax_x+1+iy, nmax_y, nmax_z+ix+1 ], 8)
    tmp_rtmp = rtmp[1::4]
    tmp_itmp = itmp[1::4]
    for ix in range(nmax_z):
        for iy in range(nmax_x):
            assert_tol(tmp_rtmp[iy*nmax_z + ix] - py_recip_space[0, nmax_x+1+iy, nmax_y, nmax_z-ix-1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_z + ix] - py_recip_space[1, nmax_x+1+iy, nmax_y, nmax_z-ix-1 ], 8)
    tmp_rtmp = rtmp[2::4]
    tmp_itmp = itmp[2::4]
    for ix in range(nmax_z):
        for iy in range(nmax_x):
            assert_tol(tmp_rtmp[iy*nmax_z + ix] - py_recip_space[0, nmax_x-1-iy, nmax_y, nmax_z-ix-1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_z + ix] - py_recip_space[1, nmax_x-1-iy, nmax_y, nmax_z-ix-1 ], 8)
    tmp_rtmp = rtmp[3::4]
    tmp_itmp = itmp[3::4]
    for ix in range(nmax_z):
        for iy in range(nmax_x):
            assert_tol(tmp_rtmp[iy*nmax_z + ix] - py_recip_space[0, nmax_x-1-iy, nmax_y, nmax_z+ix+1 ], 8)
            assert_tol(tmp_itmp[iy*nmax_z + ix] - py_recip_space[1, nmax_x-1-iy, nmax_y, nmax_z+ix+1 ], 8)


    # QUADS -------------------------------------------------------------------


    rquads = quads[:8*quad_size:]
    iquads = quads[8*quad_size::]

    tmp_rquad = rquads[0::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x+1+ix, nmax_y+1+iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))


    tmp_rquad = rquads[1::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x-1-ix, nmax_y+1+iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_rquad = rquads[2::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x-1-ix, nmax_y-1-iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_rquad = rquads[3::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x+1+ix, nmax_y-1-iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_rquad = rquads[4::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x+1+ix, nmax_y+1+iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))


    tmp_rquad = rquads[5::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x-1-ix, nmax_y+1+iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_rquad = rquads[6::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x-1-ix, nmax_y-1-iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_rquad = rquads[7::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[0, nmax_x+1+ix, nmax_y-1-iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix]))


    # Imaginary quads
    tmp_iquad = iquads[0::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x+1+ix, nmax_y+1+iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[1::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x-1-ix, nmax_y+1+iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[2::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x-1-ix, nmax_y-1-iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[3::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x+1+ix, nmax_y-1-iy, nmax_z+1+iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[4::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x+1+ix, nmax_y+1+iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[5::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x-1-ix, nmax_y+1+iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[6::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x-1-ix, nmax_y-1-iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    tmp_iquad = iquads[7::8]
    for iz in range(nmax_z):
        for iy in range(nmax_y):
            for ix in range(nmax_x):
                assert_tol(
                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix] - py_recip_space[1, nmax_x+1+ix, nmax_y-1-iy, nmax_z-1-iz],
                8, '{}_{}_{}_{}'.format(ix,iy,iz, tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix]))

    #assert abs(rs[0]*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "Energy from loop back over particles"
    assert abs(rs*c.internal_to_ev() - 0.917463161E1) < 10.**-3, "Energy from structure factor"






def test_ewald_energy_python_co2_2():
    """
    Test non cube domains reciprocal space
    """

    if mpi_rank > 0:
        return

    eta = 0.26506
    alpha = eta**2.
    rc = 12.
    e0 = 30.
    e1 = 40.
    e2 = 50.

    domain = md.domain.BaseDomainHalo(extent=(e0,e1,e2))
    c = md.coulomb.CoulombicEnergy(
        domain=domain,
        real_cutoff=12.,
        alpha=alpha,
        recip_cutoff=0.2667*scipy.constants.pi*2.0,
        recip_nmax=(8,11,14)
    )

    assert c.alpha == alpha, "unexpected alpha"
    assert c.real_cutoff == rc, "unexpected rc"

    data = np.load('../res/coulomb/CO2cuboid.npy')

    N = data.shape[0]

    positions = ParticleDat(npart=N, ncomp=3)
    charges = ParticleDat(npart=N, ncomp=1)

    positions[:] = data[:,0:3:]
    #positions[:, 0] -= e*0.5
    #positions[:, 1] -= e*0.5
    #positions[:, 2] -= e*0.5
    #print(np.max(positions[:,0]), np.min(positions[:,0]))
    #print(np.max(positions[:,1]), np.min(positions[:,1]))
    #print(np.max(positions[:,2]), np.min(positions[:,2]))

    charges[:, 0] = data[:,3]
    assert abs(np.sum(charges[:,0])) < 10.**-13, "total charge not zero"

    rs = c.evaluate_lr(positions=positions, charges=charges)

    assert abs(rs*c.internal_to_ev() - 0.3063162184E+02) < 10.**-3, "Energy from structure factor"

    return
    py_recip_space = np.load('../res/co2_recip_space.npy')


    nmax_x = c._vars['nmax_vec'][0]
    nmax_y = c._vars['nmax_vec'][1]
    nmax_z = c._vars['nmax_vec'][2]
    recip_axis_len = c._vars['recip_axis_len'].value
    recip_vec = c._vars['recip_vec']
    nmax_vec = c._vars['nmax_vec']
    coeff_space = c._vars['coeff_space']
    max_recip = c._vars['max_recip'].value
    alpha = c._vars['alpha'].value
    ivolume = c._vars['ivolume']
    recip_space = c._vars['recip_space_kernel']
    nkmax = c._vars['recip_axis_len'].value
    nkaxis = nkmax


    axes_size = 12*nkaxis
    axes = recip_space[0:axes_size:].view()
    plane_size = 4*nmax_x*nmax_y + 4*nmax_y*nmax_z + 4*nmax_z*nmax_x
    planes = recip_space[axes_size:axes_size+plane_size*2:].view()
    quad_size = nmax_x*nmax_y*nmax_z
    quad_start = axes_size+plane_size*2
    quads = recip_space[quad_start:quad_start+quad_size*16].view()









