from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

from ppmd import *
from ppmd.coulomb.fmm import *
from scipy.special import sph_harm, lpmv
import time


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier()
DEBUG = True

def spherical(xyz):
    sph = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    # r
    sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
    # polar angle
    sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
    # longitude angle
    sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

    #print(xyz, sph)

    return sph

def test_fmm_init_1():

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    fmm = PyFMM(domain=A.domain, N=1000, eps=10.**-2)
    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)


    # A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.P[:] = utility.lattice.cubic_lattice((ncubeside, ncubeside, ncubeside),
                                           (E, E, E))#[0,:]


    # perturb the positions away from the cube centers
    max_dev = 0.4*E/ncubeside
    A.P[:] += rng.uniform(low=-1. * max_dev, high=max_dev, size=(N,3))#[0,:]

    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))#[0,:]

    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias
    A.scatter_data_from(0)

    #print(A.npart_local, A.P[:A.npart_local:,:])

    #plot_spheres.draw_points(A.P[:A.npart_local:])

    #print("cube_side_len", ncubeside, "extent", E)
    #print("ncomp", fmm.L)
    #print("npart_local", A.npart_local)
    #print("N", N)
    ncomp = fmm.L**2
    #t0 = time.time()
    fmm._compute_cube_contrib(A.P, A.Q)
    #print(time.time() - t0)
    #print(np.sum(fmm.entry_data[:]))

    extent = A.domain.extent[:]
    pcells = np.zeros(A.npart_local, dtype='int')
    cube_ilen = 2**(fmm.R - 1) / extent[:]
    cube_half_len = 0.5*extent[:] / (2**(fmm.R - 1))

    shift_pos = A.P[:] + 0.5 * extent[:]
    shift_pos[:,0] = shift_pos[:,0] * cube_ilen[0]
    shift_pos[:,1] = shift_pos[:,1] * cube_ilen[1]
    shift_pos[:,2] = shift_pos[:,2] * cube_ilen[2]

    shift_pos = np.array(shift_pos, dtype='int')
    cube_centers = np.zeros(shift_pos.shape)

    cube_centers[:, 0] = (shift_pos[:, 0] * 2 + 1) * cube_half_len[0] -\
                         0.5*extent[0]
    cube_centers[:, 1]	 = (shift_pos[:, 1] * 2 + 1) * cube_half_len[1] -\
                         0.5*extent[1]
    cube_centers[:, 2] = (shift_pos[:, 2] * 2 + 1) * cube_half_len[2] -\
                         0.5*extent[2]
    shift_pos[:,0] -= fmm.entry_data.local_offset[2]
    shift_pos[:,1] -= fmm.entry_data.local_offset[1]
    shift_pos[:,2] -= fmm.entry_data.local_offset[0]


    lsize = fmm.entry_data.local_size
    for px in range(A.npart_local):
        pcells[px] = shift_pos[px, 0] + lsize[2]*(shift_pos[px, 1] +
                     lsize[1]*shift_pos[px, 2])

        cube_centers[px,:] = A.P[px,:] - cube_centers[px,:]


    sph = spherical(cube_centers[:A.npart_local:,:])

    #print("sph: radius:", sph[:, 0], " cos(theta):", np.cos(sph[:, 1]),
    #      "sin(phi)", np.sin(sph[:,2]))


    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + ncomp

    for px in range(A.npart_local):
        for lx in range(fmm.L):
            scipy_sph = sph_harm(range(0, lx+1), lx, sph[px,2], sph[px,1])
            scipy_sph = [A.Q[px] * math.sqrt(4.*math.pi/(2.*lx + 1.)) * \
                         sx for sx in scipy_sph]

            rhol = sph[px, 0]**lx
            #print(60*"-")

            # the negative m values will never match scipy as we use P^|m|_l
            for mxi, mx in enumerate(range(0, -1*lx-1, -1)):

                scipy_real = scipy_sph[mxi].real * rhol
                scipy_imag = scipy_sph[mxi].imag * rhol

                ppmd_real = fmm.entry_data[shift_pos[px,2], shift_pos[px, 1],
                                           shift_pos[px, 0], re_lm(lx, mx)]
                ppmd_imag = fmm.entry_data[shift_pos[px,2], shift_pos[px, 1],
                                           shift_pos[px, 0], im_lm(lx, mx)]

                assert abs(scipy_real - ppmd_real) < 10.**-13,\
                    "pos re fail (m,l) {} {} px {}". format(lx, mx, px)
                assert abs(scipy_imag - ppmd_imag) < 10.**-13,\
                    "pos im fail (m,l) {} {} px {}". format(lx, mx, px)



            # test the negative values
            scipy_p = A.Q[px] * lpmv(range(1, lx+1), lx, np.cos(sph[px, 1]))
            for mxi, mx in enumerate(range(1, lx)):

                re_exp = np.cos(-1.*mx*sph[px, 2])
                im_exp = np.sin(-1.*mx*sph[px, 2])


                val = math.sqrt(math.factorial(
                    lx - abs(mx))/math.factorial(lx + abs(mx)))
                val *= scipy_p[mxi] * rhol

                scipy_real = re_exp * val
                scipy_imag = im_exp * val


                ppmd_real = fmm.entry_data[shift_pos[px,2], shift_pos[px, 1],
                                           shift_pos[px, 0], re_lm(lx, mx)]
                ppmd_imag = fmm.entry_data[shift_pos[px,2], shift_pos[px, 1],
                                           shift_pos[px, 0], im_lm(lx, mx)]

                assert abs(scipy_real - ppmd_real) < 10.**-13,\
                    "neg re fail {} {}". format(lx, mx)
                assert abs(scipy_imag - ppmd_imag) < 10.**-13,\
                    "neg im fail {} {}". format(lx, mx)



def test_fmm_init_2():

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    fmm = PyFMM(domain=A.domain, N=1000, eps=10.**-3)
    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    # A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.P[:] = utility.lattice.cubic_lattice((ncubeside, ncubeside, ncubeside),
                                           (E, E, E))#[0,:]

    # perturb the positions away from the cube centers
    max_dev = 0.0*E/ncubeside
    A.P[:] += rng.uniform(low=-1. * max_dev, high=max_dev, size=(N,3))#[0,:]

    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))#[0,:]

    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias

    # override random charges
    # A.Q[:] = 1.0

    ind = np.logical_and(A.P[:,0] < -2.5, A.P[:,1] < -2.5)
    ind = np.logical_and(A.P[:,2] < -2.5, ind)



    ind2 = np.array([ 0,  1,  8,  9, 64, 65, 72, 73])

    for px in range(8):
        print(px, A.P[ind2[px],:])

    print(np.nonzero(ind))

    # create a dipole moment




    A.Q[:] = 0.
    A.Q[0] = 1.
    A.Q[1] = 0.
    A.Q[8] = 0.
    A.Q[9] = 0.
    A.Q[64] = -0.
    A.Q[65] = -0.
    A.Q[72] = -0.
    A.Q[73] = -0.

    #A.P[0,:] = [-3.75, -4.375, -3.75]
    #A.P[73,:] = [-3.75, -3.125, -3.75]



    pcell = A.P[ind, :]
    qcell = A.Q[ind]

    A.scatter_data_from(0)

    ncomp = fmm.L**2
    fmm._compute_cube_contrib(A.P, A.Q)

    pi = math.pi
    alpha_beta = (
        (1.25 * pi, 0.75 * pi),
        (1.75 * pi, 0.75 * pi),
        (0.75 * pi, 0.75 * pi),
        (0.25 * pi, 0.75 * pi),
        (1.25 * pi, 0.25 * pi),
        (1.75 * pi, 0.25 * pi),
        (0.75 * pi, 0.25 * pi),
        (0.25 * pi, 0.25 * pi)
    )

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + fmm.L**2

    # can check the positive m values match scipy's version for ylm
    for cx, child in enumerate(alpha_beta):
        for lx in range(fmm.L):
            mval = list(range(0, lx+1))

            scipy_sph = math.sqrt(4.*math.pi/(2.*lx + 1.)) * \
                        sph_harm(mval, lx, child[0], child[1])

            for mxi, mx in enumerate(mval):
                assert abs(scipy_sph[mxi].real - \
                           fmm._yab[cx, re_lm(lx, mx)]) < 10.**-15, \
                'real ylm error l {} m {}'.format(lx, mx)

                assert abs(scipy_sph[mxi].imag - \
                           fmm._yab[cx, im_lm(lx, mx)]) < 10.**-15, \
                'imag ylm error l {} m {}'.format(lx, mx)

    # check the A_n^m coefficients.
    for nx in range(fmm.L):
        for mx in range(-1*fmm.L, fmm.L+1):
            if abs(mx) > nx:
                cval = 0.0
            else:
                cval = ((-1.0)**nx) / math.sqrt(
                    math.factorial(nx - mx)*math.factorial(nx + mx))

            assert abs(cval - fmm._a[nx, abs(mx)]) < 10.**-15, \
                "failed n, m {}, {}".format(nx, mx)


    fmm.tree_halo[fmm.R-1][2:-2:, 2:-2:, 2:-2:, :] = fmm.entry_data[:,:,:,:]
    print('\n')
    fmm._translate_m_to_m(fmm.R-1)


    #print(fmm.tree_halo[fmm.R-1][2:-2,2:-2,2:-2,1])
    #for lx in range(fmm.L):
    #    print(lx, 60*'-')
    #    for mx in range(-1*lx, lx+1):
    #        print(mx)
    #        print(fmm.tree_parent[fmm.R-1][:,:,:, re_lm(lx, mx)])













    print(60*'~')

    point = np.array((-7., -7., -7.))

    phi = A.Q[0] / np.linalg.norm(point - A.P[0, :])

    print("phi_direct", phi, '\t\t', A.P[0,:], A.Q[0])



    moments = fmm.entry_data[0,0,0,:]



    center = np.array((-4.375, -4.375, -4.375))
    disp = point - center

    disp_sph = spherical(np.reshape(disp, (1, 3)))

    print("sph", disp_sph)

    phi_sph_re = 0.
    phi_sph_im = 0.

    llimit = fmm.L

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(mx*disp_sph[0,2])
            im_exp = np.sin(mx*disp_sph[0,2])

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = 1. / (disp_sph[0,0] ** (lx+1.))

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

    print("phi_sph", phi_sph_re, '+', phi_sph_im, 'i', )

    # print(moments[:llimit**2:])
    print(60*'~')



















    point = np.array((-7., -7., -7.))

    phi = 0.
    charge_total = 0.

    for px in range(8):
        print("Q:", qcell[px], "\tPOS:", pcell[px,:])
        phi += qcell[px] / np.linalg.norm(point - pcell[px])
        charge_total += qcell[px, 0]





    moments = fmm.tree_parent[fmm.R-1][0,0,0,:]


    center = np.array((-3.75, -3.75, -3.75))
    disp = point - center

    disp_sph = spherical(np.reshape(disp, (1, 3)))

    phi = 1.0/disp_sph[0,0]

    print("phi_direct", phi)
    print("sph", disp_sph)
    print("charge_total", charge_total)


    phi_sph_re = 0.
    phi_sph_im = 0.

    for lx in range(llimit):
        print(lx,  '----------------')
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        irad = 1. / (disp_sph[0,0] ** (lx+1.))



        for mxi, mx in enumerate(mrange2):


            re_exp = np.cos(mx*disp_sph[0,2])
            im_exp = np.sin(mx*disp_sph[0,2])

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi] * irad

            scipy_real = re_exp * val
            scipy_imag = im_exp * val

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

            print(mx, ppmd_mom_re, ppmd_mom_im)


    print("phi_sph", phi_sph_re, '+', phi_sph_im, 'i')



    # print(moments[:llimit**2:])
    print(60*'~')
    print("ERR:", abs(phi_sph_re - phi))
    print(60*'~')






























