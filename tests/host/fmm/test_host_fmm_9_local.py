from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

#from ppmd_vis import plot_spheres

import itertools


from ppmd import *
from ppmd.coulomb.fmm import *
from scipy.special import sph_harm, lpmv
import time


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = False

def spherical(xyz):
    if type(xyz) is tuple:
        sph = np.zeros(3)
        xy = xyz[0]**2 + xyz[1]**2
        # r
        sph[0] = np.sqrt(xy + xyz[2]**2)
        # polar angle
        sph[1] = np.arctan2(np.sqrt(xy), xyz[2])
        # longitude angle
        sph[2] = np.arctan2(xyz[1], xyz[0])

    else:
        sph = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        # r
        sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
        # polar angle
        sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
        # longitude angle
        sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

    return sph

def compute_phi(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):
            #print('mx', mx)

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

    return phi_sph_re, phi_sph_im



def compute_phi_local(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(mx*disp_sph[0,2])
            im_exp = np.sin(mx*disp_sph[0,2])

            #print('mx', mx, im_exp)

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = disp_sph[0,0] ** (lx)

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

    return phi_sph_re, phi_sph_im

@pytest.mark.skipif("MPISIZE > 1")
def test_fmm_init_2_1():

    E = 20.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-3

    fmm = PyFMM(domain=A.domain, r=3, eps=eps, free_space=True)

    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    N = 100
    A.npart = N

    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.U = data.ParticleDat(ncomp=1)
    A.F = data.ParticleDat(ncomp=3)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))

    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))

    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias

    #A.Q[:] = 1.0
    #A.Q[0:-2] = 0.0

    #A.P[2,:] = (-7.5, -7.5,  2.5)
    #A.Q[2, 0] = -1.

    #A.P[0,:] = (-6., 2., -1.)
    #A.P[1,:] = (5.000000001, 5.000000001, -4.5)
    #A.Q[0, 0] = 1.0
    #A.Q[1, 0] = -1.0

    A.scatter_data_from(0)

    maxt = 5
    for tx in itertools.product(
            range(-1*maxt, maxt+1), range(-1*maxt, maxt+1),
            range(-1*maxt, maxt+1)):

        dispt = (tx[0]*E, tx[1]*E, tx[2]*E)
        dispt_sph = spherical(dispt)
        dispt_fmm = fmm._image_to_sph(tx)
        assert abs(dispt_sph[0] - dispt_fmm[0]) < 10.**-16, "bad radius"
        assert abs(dispt_sph[2] - dispt_fmm[1]) < 10.**-16, "bad phi"
        assert abs(dispt_sph[1] - dispt_fmm[2]) < 10.**-16, "bad theta"

    fmm._check_aux_dat(A.P)
    fmm._compute_cube_contrib(A.P, A.Q, A._fmm_cell)

    # compute potential energy to point across all charges directly
    P2 = data.PositionDat(npart=N, ncomp=3)
    Q2 = data.ParticleDat(npart=N, ncomp=1)    
    U2 = data.ParticleDat(npart=N, ncomp=1)
    P2[:,:] = A.P[:N:,:]
    Q2[:,:] = A.Q[:N:,:]
    phi_ga = data.ScalarArray(ncomp=1, dtype=ctypes.c_double)

    src = """
    const double d0 = P.j[0] - P.i[0];
    const double d1 = P.j[1] - P.i[1];
    const double d2 = P.j[2] - P.i[2];
    const double contrib = Q.i[0] * Q.j[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    phi[0] += 0.5 * contrib;
    U.i[0] += contrib;
    """
    phi_kernel = kernel.Kernel('all_to_all_phi', src,
                               headers=(kernel.Header('math.h'),))


    phi_loop = pairloop.AllToAllNS(kernel=phi_kernel,
                                   dat_dict={'P': P2(access.READ),
                                             'Q': Q2(access.READ),
                                             'phi': phi_ga(access.INC_ZERO),
                                             'U': U2(access.INC_ZERO)})
    phi_loop.execute()
    

    if DEBUG:
        phi_py = 0.0
        for ix in range(N):
            for jx in range(ix+1, N):
                r2 = np.linalg.norm(A.P[jx, :] - A.P[ix,:])
                phi_py += A.Q[ix,0] * A.Q[jx,0] / r2

        print("Direct", phi_ga[0], phi_py)
    
    extent = A.domain.extent
    cube_ilen = 2**(fmm.R - 1) / extent[:]
    cube_half_len = 0.5*extent[:] / (2**(fmm.R - 1))
    shift_pos = A.P[:] + 0.5 * extent[:]
    shift_pos[:,0] = shift_pos[:,0] * cube_ilen[0]
    shift_pos[:,1] = shift_pos[:,1] * cube_ilen[1]
    shift_pos[:,2] = shift_pos[:,2] * cube_ilen[2]
    shift_pos = np.array(shift_pos, dtype='int') 

    phi_local = fmm._compute_local_interaction(A.P, A.Q)
    #print("phi_local:", phi_local)
    #print("------------------------------------------------")
    phi_local_pair = fmm._compute_local_interaction_pairloop(A.P, A.Q)
    #print("correct  :", phi_local)
    
    assert abs(phi_local-phi_local_pair) < 10.**-12

    level = fmm.R - 1

    phi_sph_re = 0.0
    phi_sph_im = 0.0

    lsize = fmm.tree_plain[level][:,:,:,0].shape

    sep = A.domain.extent[0] / float(2.**level)
    start_point = -0.5*E + 0.5*sep


    for px in range(N):
        point = A.P[px, :]
        charge = A.Q[px, 0]
        if DEBUG:
            print(px, point, charge, shift_pos[px, :])
        cubex = shift_pos[px,:]

        for momx in itertools.product(range(lsize[0]),
                                      range(lsize[1]),
                                      range(lsize[2])):
            
            dist = (
                (momx[2] - cubex[0])**2 +
                (momx[1] - cubex[1])**2 +
                (momx[0] - cubex[2])**2
            )
            if dist > 3:
            
                center = np.array(
                    (start_point + (momx[2])*sep,
                     start_point + (momx[1])*sep,
                     start_point + (momx[0])*sep))

                disp = point - center
                moments = fmm.tree_halo[level][
                          momx[0]+2, momx[1]+2, momx[2]+2, :]
                disp_sph = spherical(np.reshape(disp, (1, 3)))

                phi_sph_re1, phi_sph_im1 = compute_phi(fmm.L, moments,
                                                       disp_sph)
                phi_sph_re += phi_sph_re1 * charge * 0.5
                phi_sph_im += phi_sph_im1 * charge * 0.5

    if DEBUG:
        print("direct:", phi_ga[0])
        print("multipole:", phi_sph_re + phi_local, "local:", phi_local, 
        "expansions:", phi_sph_re)
    assert abs(phi_ga[0] - phi_sph_re - phi_local) < eps, "bad real part"


    phi_py2 = fmm(A.P, A.Q, forces=A.F, potential=A.U)
    
    for px in range(A.npart_local):
        assert abs(U2[px,0] - A.U[px,0]) < eps



    fmm.free()













