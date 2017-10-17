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
            scipy_sph = [math.sqrt(4.*math.pi/(2.*lx + 1.)) * sx for sx in \
                         scipy_sph]

            #print(60*"-")

            # the negative m values will never match scipy as we use P^|m|_l
            for mxi, mx in enumerate(range(0, lx+1)):

                scipy_real = scipy_sph[mxi].real
                scipy_imag = scipy_sph[mxi].imag

                ppmd_real = fmm.entry_data[shift_pos[px,2], shift_pos[px, 1],
                                           shift_pos[px, 0], re_lm(lx, mx)]
                ppmd_imag = fmm.entry_data[shift_pos[px,2], shift_pos[px, 1],
                                           shift_pos[px, 0], im_lm(lx, mx)]

                assert abs(scipy_real - ppmd_real) < 10.**-13,\
                    "pos re fail (m,l) {} {} px {}". format(lx, mx, px)
                assert abs(scipy_imag - ppmd_imag) < 10.**-13,\
                    "pos im fail (m,l) {} {} px {}". format(lx, mx, px)


                #print("l {} m {} sci {} {} ppmd {} {}".format(
                #    lx, mx, scipy_real, scipy_imag, ppmd_real, ppmd_imag
                #))


            #print(60*"=")

            # test the negative values
            scipy_p = lpmv(range(1, lx+1), lx, np.cos(sph[px, 1]))
            for mxi, mx in enumerate(range(-1, -1*lx - 1,-1)):

                re_exp = np.cos(mx*sph[px, 2])
                im_exp = np.sin(mx*sph[px, 2])

                #print("exp ", mx, re_exp, im_exp)


                val = math.sqrt(math.factorial(
                    lx - abs(mx))/math.factorial(lx + abs(mx)))
                val *= scipy_p[mxi]

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


                #print("l {} m {} sci {} {} ppmd {} {}".format(
                #    lx, mx, scipy_real, scipy_imag, ppmd_real, ppmd_imag
                #))

def test_fmm_init_2():

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


    ncomp = fmm.L**2
    fmm._compute_cube_contrib(A.P, A.Q)


    fmm._translate_m_to_m(fmm.R-1)





































