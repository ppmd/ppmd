from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools


from ppmd import *
from ppmd.coulomb.fmm import *
from scipy.special import sph_harm, lpmv
import time


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True

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
def test_fmm_init_3_1():

    offset = (20., 0., 0.)

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2

    fmm = PyFMM(domain=A.domain, N=100, eps=eps, free_space=True)

    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    
    #N = 2
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    
    
    A.P[:] = utility.lattice.cubic_lattice((ncubeside, ncubeside, ncubeside),
                                           (E, E, E))
    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))
    
    #A.P[0, :] = (-3.75, -3.75, -3.75)
    #A.P[1, :] = (3.75, 3.75, 3.75)

    A.Q[0,0] = 1.0
    A.Q[1,0] = 1.0


    bias = np.sum(A.Q[:])/N

    A.scatter_data_from(0)

    # compute potential energy to point across all charges directly
    P2 = data.PositionDat(npart=N, ncomp=3)
    Q2 = data.ParticleDat(npart=N, ncomp=1)    
    P2[:,:] = A.P[:N:,:]
    Q2[:,:] = A.Q[:N:,:]
    phi_ga = data.ScalarArray(ncomp=1, dtype=ctypes.c_double)
    src = """
    const double d0 = P.j[0] - P.i[0];
    const double d1 = P.j[1] - P.i[1];
    const double d2 = P.j[2] - P.i[2];
    phi[0] += 0.5 * Q.i[0] * Q.j[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """
    phi_kernel = kernel.Kernel('all_to_all_phi', src,
                               headers=(kernel.Header('math.h'),))


    phi_loop = pairloop.AllToAllNS(kernel=phi_kernel,
                                   dat_dict={'P': P2(access.READ),
                                             'Q': Q2(access.READ),
                                             'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()


    #print("fmm.R", fmm.R)
    fmm._compute_cube_contrib(A.P, A.Q)

    for level in range(fmm.R - 1, 0, -1):

        fmm._translate_m_to_m(level)

        #print("HALO", level)
        #print(fmm.tree_halo[level][:,:,:,0])

        fmm._translate_m_to_l(level)
        #fmm._fine_to_coarse(level)

    fmm.tree_parent[1][:] = 0.0

    for level in range(1, fmm.R):
        
        fmm.tree_parent[level][:] = 0.0
        fmm._coarse_to_fine(level)
        fmm._translate_l_to_l(level)

        #print("PLAIN", level)
        #print(fmm.tree_plain[level][:,:,:,0])

    fmm._compute_local_interaction(A.P, A.Q)
    phi_local = fmm.particle_phi[0]

    extent = A.domain.extent
    cube_ilen = 2**(fmm.R - 1) / extent[:]
    cube_half_len = 0.5*extent[:] / (2**(fmm.R - 1))
    shift_pos = A.P[:] + 0.5 * extent[:]
    shift_pos[:,0] = shift_pos[:,0] * cube_ilen[0]
    shift_pos[:,1] = shift_pos[:,1] * cube_ilen[1]
    shift_pos[:,2] = shift_pos[:,2] * cube_ilen[2]
    shift_pos = np.array(shift_pos, dtype='int') 

    
    phi_py = 0.0
    for px in range(N):
        s = shift_pos[px]
        phi_py += fmm.tree_plain[fmm.R - 1][s[2],s[1],s[0],0] * \
            A.Q[px, 0] * 0.5
    
    phi_fmm = phi_local + phi_py

    if DEBUG:
        print("phi_local", phi_local, "phi_py", phi_py)
        print("direct:", phi_ga[0], "phi_fmm", phi_fmm)
        print("ERROR:", abs(phi_ga[0] - phi_fmm))
    assert abs(phi_ga[0] - phi_fmm) < 10.**-10


@pytest.mark.skipif("True")
def test_fmm_init_3_2():

    offset = (20., 0., 0.)

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2

    fmm = PyFMM(domain=A.domain, N=1000, eps=eps, free_space=True)

    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    
    #N = 2
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    
    
    A.P[:] = utility.lattice.cubic_lattice((ncubeside, ncubeside, ncubeside),
                                           (E, E, E))
    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))

    
    bias = np.sum(A.Q[:])/N

    A.scatter_data_from(0)

    # compute potential energy to point across all charges directly
    P2 = data.PositionDat(npart=N, ncomp=3)
    Q2 = data.ParticleDat(npart=N, ncomp=1)    
    P2[:,:] = A.P[:N:,:]
    Q2[:,:] = A.Q[:N:,:]
    phi_ga = data.ScalarArray(ncomp=1, dtype=ctypes.c_double)
    src = """
    const double d0 = P.j[0] - P.i[0];
    const double d1 = P.j[1] - P.i[1];
    const double d2 = P.j[2] - P.i[2];
    phi[0] += 0.5 * Q.i[0] * Q.j[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """
    phi_kernel = kernel.Kernel('all_to_all_phi', src,
                               headers=(kernel.Header('math.h'),))


    phi_loop = pairloop.AllToAllNS(kernel=phi_kernel,
                                   dat_dict={'P': P2(access.READ),
                                             'Q': Q2(access.READ),
                                             'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()


    fmm._compute_cube_contrib(A.P, A.Q)

    for level in range(fmm.R - 1, 0, -1):

        fmm._translate_m_to_m(level)

        #print("HALO", level)
        #print(fmm.tree_halo[level][:,:,:,0])

        fmm._translate_m_to_l(level)
        fmm._fine_to_coarse(level)

    fmm.tree_parent[1][:] = 0.0

    for level in range(1, fmm.R):
        
        fmm._coarse_to_fine(level)
        fmm._translate_l_to_l(level)

        #print("PLAIN", level)
        #print(fmm.tree_plain[level][:,:,:,0])

    fmm._compute_local_interaction(A.P, A.Q)
    phi_local = fmm.particle_phi[0]

    extent = A.domain.extent
    cube_ilen = 2**(fmm.R - 1) / extent[:]
    cube_half_len = 0.5*extent[:] / (2**(fmm.R - 1))
    shift_pos = A.P[:] + 0.5 * extent[:]
    shift_pos[:,0] = shift_pos[:,0] * cube_ilen[0]
    shift_pos[:,1] = shift_pos[:,1] * cube_ilen[1]
    shift_pos[:,2] = shift_pos[:,2] * cube_ilen[2]
    shift_pos = np.array(shift_pos, dtype='int') 

    
    phi_py = 0.0
    for px in range(N):
        s = shift_pos[px]
        phi_py += fmm.tree_plain[fmm.R - 1][s[2],s[1],s[0],0] * \
            A.Q[px, 0] * 0.5
    
    phi_fmm = phi_local + phi_py

    if DEBUG:
        print("phi_local", phi_local, "phi_py", phi_py)
        print("direct:", phi_ga[0], "phi_fmm", phi_fmm)
        print("ERROR:", abs(phi_ga[0] - phi_fmm))

    assert abs(phi_ga[0] - phi_fmm) < eps


def test_fmm_init_3_3():

    offset = (20., 0., 0.)

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4

    fmm = PyFMM(domain=A.domain, N=1000, eps=eps, free_space=True)

    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    
    #N = 2
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    
    #A.P[:] = utility.lattice.cubic_lattice((ncubeside, ncubeside, ncubeside),
    #                                       (E, E, E))
    
    A.P[:] = rng.uniform(low=-0.499*E, high=0.499*E, size=(N,3))
    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))
    

    #A.P[0,:] = (-4.375 + 0.2, -4.375, -4.375)
    #A.P[1,:] = (3.125, 3.125, -1.875)

    #A.Q[0,0] = 1.
    #A.Q[1,0] = 1.

    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias


    A.scatter_data_from(0)



    # compute potential energy to point across all charges directly
    P2 = data.PositionDat(npart=N, ncomp=3)
    Q2 = data.ParticleDat(npart=N, ncomp=1)    
    P2[:,:] = A.P[:N:,:]
    Q2[:,:] = A.Q[:N:,:]
    phi_ga = data.ScalarArray(ncomp=1, dtype=ctypes.c_double)
    src = """
    const double d0 = P.j[0] - P.i[0];
    const double d1 = P.j[1] - P.i[1];
    const double d2 = P.j[2] - P.i[2];
    phi[0] += 0.5 * Q.i[0] * Q.j[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """
    phi_kernel = kernel.Kernel('all_to_all_phi', src,
                               headers=(kernel.Header('math.h'),))


    phi_loop = pairloop.AllToAllNS(kernel=phi_kernel,
                                   dat_dict={'P': P2(access.READ),
                                             'Q': Q2(access.READ),
                                             'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()


    fmm._compute_cube_contrib(A.P, A.Q)

    for level in range(fmm.R - 1, 0, -1):

        fmm._translate_m_to_m(level)

        #print("HALO", level)
        #print(fmm.tree_halo[level][:,:,:,0])

        fmm._translate_m_to_l(level)
        fmm._fine_to_coarse(level)

    fmm.tree_parent[1][:] = 0.0

    for level in range(1, fmm.R):
        
        fmm._coarse_to_fine(level)
        fmm._translate_l_to_l(level)

        #print("PLAIN", level)
        #print(fmm.tree_plain[level][:,:,:,0])

    fmm._compute_local_interaction(A.P, A.Q)
    phi_local = fmm.particle_phi[0]
    
    phi_py = fmm._compute_cube_extraction(A.P, A.Q)

    
    phi_fmm = phi_local + phi_py

    if DEBUG:
        print("phi_local", phi_local, "phi_py", phi_py)
        print("direct:", phi_ga[0], "phi_fmm", phi_fmm)
        print("ERROR:", abs(phi_ga[0] - phi_fmm))

    #assert abs(phi_ga[0] - phi_fmm) < eps





















