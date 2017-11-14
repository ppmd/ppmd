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

def test_fmm_init_1():

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    fmm = PyFMM(domain=A.domain, r=4, eps=10.**-2, free_space=True)
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





def test_fmm_init_2():

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4

    fmm = PyFMM(domain=A.domain, r=4, eps=eps, free_space=True)
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
    max_dev = 0.45*E/ncubeside
    A.P[:] += rng.uniform(low=-1. * max_dev, high=max_dev, size=(N,3))#[0,:]

    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))#[0,:]

    bias = np.sum(A.Q[:])/N

    A.Q[:] -= bias

    # override random charges
    # A.Q[:] = 1.0

    ind = np.logical_and(A.P[:,0] < -2.5, A.P[:,1] < -2.5)
    ind = np.logical_and(A.P[:,2] < -2.5, ind)
    ind2 = np.array([ 0,  1,  8,  9, 64, 65, 72, 73])

    #for px in range(8):
    #    print(px, A.P[ind2[px],:])

    #print(np.nonzero(ind))

    # create a dipole moment


    #A.Q[:] = 0.
    #A.Q[0] = 1.
    #A.Q[1] = 1.
    #A.Q[8] = 1.
    #A.Q[9] = 1.
    #A.Q[64] = -0.
    #A.Q[65] = -0.
    #A.Q[72] = -0.
    #A.Q[73] = -0.

    #A.P[0,:] = [-3.75, -4.375, -3.75]
    #A.P[73,:] = [-3.75, -3.125, -3.75]

    #A.P[0, :] = (-3.75-eps, -3.75-eps, -3.75-eps)


    pcell = A.P[ind, :]
    qcell = A.Q[ind]

    A.scatter_data_from(0)

    ncomp = fmm.L**2
    fmm._compute_cube_contrib(A.P, A.Q)

    pi = math.pi
    alpha_beta = (
        (1.25 * pi,-1./math.sqrt(3.)),
        (1.75 * pi,-1./math.sqrt(3.)),
        (0.75 * pi,-1./math.sqrt(3.)),
        (0.25 * pi,-1./math.sqrt(3.)),
        (1.25 * pi, 1./math.sqrt(3.)),
        (1.75 * pi, 1./math.sqrt(3.)),
        (0.75 * pi, 1./math.sqrt(3.)),
        (0.25 * pi, 1./math.sqrt(3.))
    )

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + fmm.L**2

    # can check the positive m values match scipy's version for ylm
    for cx, child in enumerate(alpha_beta):
        for lx in range(fmm.L*2):
            mval = list(range(0, lx+1))

            scipy_sph = math.sqrt(4.*math.pi/(2.*lx + 1.)) * \
                        sph_harm(mval, lx, child[0], math.acos(child[1]))

            for mxi, mx in enumerate(mval):
                assert abs(scipy_sph[mxi].real - \
                           fmm._yab[cx, re_lm(lx, mx)]) < 10.**-15, \
                'real ylm error l {} m {}'.format(lx, mx)

                assert abs(scipy_sph[mxi].imag - \
                           fmm._yab[cx, (fmm.L*2)**2 + re_lm(lx, mx)]) < \
                       10.**-15, 'imag ylm error l {} m {}'.format(lx, mx)

    # check the A_n^m coefficients.
    for nx in range(fmm.L*2):
        for mx in range(-2*fmm.L, 2*fmm.L+1):
            if abs(mx) > nx:
                cval = 0.0
            else:
                cval = ((-1.0)**nx) / math.sqrt(
                    math.factorial(nx - mx)*math.factorial(nx + mx))

            assert abs(cval - fmm._a[nx, fmm.L*2 + mx]) < 10.**-15, \
                "failed n, m {}, {}".format(nx, mx)




    #print(fmm.tree_halo[fmm.R-1][2:-2,2:-2,2:-2,1])
    #for lx in range(fmm.L):
    #    print(lx, 60*'-')
    #    for mx in range(-1*lx, lx+1):
    #        print(mx)
    #        print(fmm.tree_parent[fmm.R-1][:,:,:, re_lm(lx, mx)])

    point = np.array((-15., -15., -15.))

    """
    print(60*'~')


    phi = A.Q[0] / np.linalg.norm(point - A.P[0, :])

    print("phi_direct", phi, '\t\t', A.P[0,:], A.Q[0])


    moments = fmm.entry_data[0,0,0,:]
    #print("moments", moments)


    center = np.array((-4.375, -4.375, -4.375))
    # center = A.P[0,:]


    disp = point - center

    disp_sph = spherical(np.reshape(disp, (1, 3)))

    print("sph", disp_sph)
    
    llimit = fmm.L
    phi_sph_re, phi_sph_im = compute_phi(llimit, moments, disp_sph)
    

    print("phi_sph", phi_sph_re, '+', phi_sph_im, 'i', )

    print("ERR:", abs(phi_sph_re - phi))
    # print(moments[:llimit**2:])
    print(60*'~')
    """


    # compute potential energy to point across all charges directly
    src = """
    const double d0 = P.i[0] - {};
    const double d1 = P.i[1] - {};
    const double d2 = P.i[2] - {};
    phi[0] += Q.i[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """.format(point[0], point[1], point[2])
    phi_kernel = kernel.Kernel('point_phi', src,
                               headers=(kernel.Header('math.h'),))
    phi_ga = data.GlobalArray(ncomp=1, dtype=ctypes.c_double)
    phi_loop = loop.ParticleLoopOMP(kernel=phi_kernel,
                                    dat_dict={'P': A.P(access.READ),
                                              'Q': A.Q(access.READ),
                                              'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()


    for level in range(fmm.R - 1, 0, -1):
        #if MPIRANK == 0:
        #    print(level)
        phi_sph_re = 0.0
        phi_sph_im = 0.0

        lsize = fmm.tree[level].parent_local_size

        #print(MPIRANK, lsize, fmm.tree[level-1].parent_local_size)

        if lsize is not None:

            fmm._translate_m_to_m(level)
            fmm._fine_to_coarse(level)

            parent_shape = fmm.tree_plain[level][:,:,:,0].shape

            sep = A.domain.extent[0] / float(2.**(level - 1.))
            start_point = -0.5*E + 0.5*sep

            offset = fmm.tree[level].local_grid_offset

            if lsize is not None:
                for momx in itertools.product(range(parent_shape[0]//2),
                                              range(parent_shape[1]//2),
                                              range(parent_shape[2]//2)):

                    center = np.array(
                        (start_point + (offset[2]//2 + momx[2])*sep,
                        start_point + (offset[1]//2 + momx[1])*sep,
                        start_point + (offset[0]//2 + momx[0])*sep))
                    disp = point - center
                    moments = fmm.tree_parent[level][
                              momx[0], momx[1], momx[2], :]
                    disp_sph = spherical(np.reshape(disp, (1, 3)))

                    phi_sph_re1, phi_sph_im1 = compute_phi(fmm.L, moments,
                                                           disp_sph)
                    phi_sph_re += phi_sph_re1
                    phi_sph_im += phi_sph_im1

        if level < fmm.R-1:
            last_re = red_re
        else:
            last_re = 0.0

        red_re = mpi.all_reduce(np.array((phi_sph_re)))
        red_im = mpi.all_reduce(np.array((phi_sph_im)))

        red_re = abs(red_re - phi_ga[0])
        red_im = abs(red_im)

        # print(moments[:llimit**2:])
        if MPIRANK == 0 and DEBUG:
            print(60*'~')
            print("ERR RE:", red_re)
            print("ERR IM:", red_im)
            print(60*'~')

        assert red_im < 10.**-15, "bad imaginary part"
        assert red_re > last_re, "Errors do not get better as level -> 0"
        assert red_re < eps, "error did not meet tol"


def test_fmm_init_3():

    E = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-3

    fmm = PyFMM(domain=A.domain, N=1000, eps=eps)
    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))


    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))#[0,:]

    bias = np.sum(A.Q[:])/N

    A.Q[:] -= bias

    A.scatter_data_from(0)

    fmm._compute_cube_contrib(A.P, A.Q)

    pi = math.pi

    point = np.array((-15., -15., -15.))

    # compute potential energy to point across all charges directly
    src = """
    const double d0 = P.i[0] - {};
    const double d1 = P.i[1] - {};
    const double d2 = P.i[2] - {};
    phi[0] += Q.i[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """.format(point[0], point[1], point[2])
    phi_kernel = kernel.Kernel('point_phi', src,
                               headers=(kernel.Header('math.h'),))
    phi_ga = data.GlobalArray(ncomp=1, dtype=ctypes.c_double)
    phi_loop = loop.ParticleLoopOMP(kernel=phi_kernel,
                                    dat_dict={'P': A.P(access.READ),
                                              'Q': A.Q(access.READ),
                                              'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()


    for level in range(fmm.R - 1, 0, -1):
        #if MPIRANK == 0:
        #    print(level)
        phi_sph_re = 0.0
        phi_sph_im = 0.0

        lsize = fmm.tree[level].parent_local_size

        #print(MPIRANK, lsize, fmm.tree[level-1].parent_local_size)

        if lsize is not None:
            fmm._translate_m_to_m(level)
            fmm._fine_to_coarse(level)

            parent_shape = fmm.tree_plain[level][:,:,:,0].shape

            sep = A.domain.extent[0] / float(2.**(level - 1.))
            start_point = -0.5*E + 0.5*sep

            offset = fmm.tree[level].local_grid_offset

            if lsize is not None:
                for momx in itertools.product(range(parent_shape[0]//2),
                                              range(parent_shape[1]//2),
                                              range(parent_shape[2]//2)):

                    center = np.array(
                        (start_point + (offset[2]//2 + momx[2])*sep,
                        start_point + (offset[1]//2 + momx[1])*sep,
                        start_point + (offset[0]//2 + momx[0])*sep))
                    disp = point - center
                    moments = fmm.tree_parent[level][
                              momx[0], momx[1], momx[2], :]
                    disp_sph = spherical(np.reshape(disp, (1, 3)))

                    phi_sph_re1, phi_sph_im1 = compute_phi(fmm.L, moments,
                                                           disp_sph)
                    phi_sph_re += phi_sph_re1
                    phi_sph_im += phi_sph_im1

        if level < fmm.R-1:
            last_re = red_re
        else:
            last_re = 0.0

        red_re = mpi.all_reduce(np.array((phi_sph_re)))
        red_im = mpi.all_reduce(np.array((phi_sph_im)))

        red_re = abs(red_re - phi_ga[0])
        red_im = abs(red_im)

        # print(moments[:llimit**2:])
        if MPIRANK == 0 and DEBUG:
            print(60*'~')
            print("ERR RE:", red_re)
            print("ERR IM:", red_im)
            print(60*'~')

        assert red_im < 10.**-15, "bad imaginary part"
        assert red_re > last_re, "Errors do not get better as level -> 0"
        assert red_re < eps, "error did not meet tol"




@pytest.fixture(
    scope="module",
    params=(
            (15.,0., 0.),
            (-15.,0., 0.),
            (15.,15., 0.),
            (15.,-15., 0.),
            (15.,-15., 15.)
    )
)
def offset(request):
    return request.param


def test_fmm_init_4():
    offset = (30., 30., 30.)

    E = 20.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2

    fmm = PyFMM(domain=A.domain, r=5, eps=eps, free_space=True)
    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    #N = 1
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))

    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))

    bias = np.sum(A.Q[:])/N

    # A.Q[:] -= bias

    #A.P[0,:] = 0.0
    #A.Q[0] = 1.0
    #A.Q[1] = -1.0

    #A.P[0, :] = (0.0, 4.999999999999, 0.)
    #A.P[1, :] = (0.0, -4.999999999999, 0.)

    #A.P[0, :] = (4.999999999999, 0., 0.)
    #A.P[1, :] = (-4.999999999999, 0., 0.)

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

    fmm._compute_cube_contrib(A.P, A.Q)

    point = np.array(offset, dtype='float64')

    # compute potential energy to point across all charges directly
    src = """
    const double d0 = P.i[0] - POINT[0];
    const double d1 = P.i[1] - POINT[1];
    const double d2 = P.i[2] - POINT[2];
    phi[0] += Q.i[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """.format(point[0], point[1], point[2])
    phi_kernel = kernel.Kernel('point_phi', src,
                               headers=(kernel.Header('math.h'),))
    phi_ga = data.GlobalArray(ncomp=1, dtype=ctypes.c_double)
    point_sa = data.ScalarArray(ncomp=3, dtype=ctypes.c_double)
    point_sa[:] = point
    phi_loop = loop.ParticleLoopOMP(kernel=phi_kernel,
                                    dat_dict={'P': A.P(access.READ),
                                              'Q': A.Q(access.READ),
                                              'POINT': point_sa(access.READ),
                                              'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()

    print("\n")

    for level in range(fmm.R - 1, 0, -1):
        #if MPIRANK == 0:
        #    print(level)
        phi_sph_re = 0.0
        phi_sph_im = 0.0

        lsize = fmm.tree[level].parent_local_size

        #print(MPIRANK, lsize, fmm.tree[level-1].parent_local_size)

        if lsize is not None:
            fmm._translate_m_to_m(level)
            fmm._fine_to_coarse(level)

            parent_shape = fmm.tree_plain[level][:,:,:,0].shape

            sep = A.domain.extent[0] / float(2.**(level - 1.))
            start_point = -0.5*E + 0.5*sep

            offset = fmm.tree[level].local_grid_offset

            if lsize is not None:
                for momx in itertools.product(range(parent_shape[0]//2),
                                              range(parent_shape[1]//2),
                                              range(parent_shape[2]//2)):

                    center = np.array(
                        (start_point + (offset[2]//2 + momx[2])*sep,
                        start_point + (offset[1]//2 + momx[1])*sep,
                        start_point + (offset[0]//2 + momx[0])*sep))
                    disp = point - center
                    moments = fmm.tree_parent[level][
                              momx[0], momx[1], momx[2], :]
                    disp_sph = spherical(np.reshape(disp, (1, 3)))

                    phi_sph_re1, phi_sph_im1 = compute_phi(fmm.L, moments,
                                                           disp_sph)
                    phi_sph_re += phi_sph_re1
                    phi_sph_im += phi_sph_im1

        if level < fmm.R-1:
            last_re = red_re
        else:
            last_re = 0.0

        red_re = mpi.all_reduce(np.array((phi_sph_re)))
        red_im = mpi.all_reduce(np.array((phi_sph_im)))

        red_re = abs(red_re - phi_ga[0])
        red_im = abs(red_im)

        # print(moments[:llimit**2:])
        if MPIRANK == 0 and DEBUG:
            print('MULTI', 60*'~')
            print("ERR RE:", red_re)
            print("ERR IM:", red_im)
            print(60*'~')

        assert red_im < 10.**-15, "bad imaginary part"
        #assert red_re >= last_re, "Errors do not get better as level -> 0"
        assert red_re < eps, "error did not meet tol"
    

    # after traversing up tree
    disp = -1. * point #  center is (0,0,0)
    disp_sph = spherical(np.reshape(disp, (1, 3)))

    # spherical harmonics needed for point

    exp_array = np.zeros(fmm.L*8 + 2, dtype=ctypes.c_double)
    p_array = np.zeros((fmm.L*2)**2, dtype=ctypes.c_double)

    def re_lm(l,m): return (l**2) + l + m

    for lx in range(fmm.L*2):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        for mxi, mx in enumerate(mrange2):
            coeff = math.sqrt(float(math.factorial(lx-abs(mx)))/
                math.factorial(lx+abs(mx)))
            p_array[re_lm(lx, mx)] = scipy_p[mxi].real*coeff

    for mxi, mx in enumerate(list(
            range(-2*fmm.L, 1)) + list(range(1, 2*fmm.L+1))
        ):

        exp_array[mxi] = np.cos(mx*disp_sph[0,2])
        exp_array[mxi + fmm.L*4 + 1] = np.sin(mx*disp_sph[0,2])

    l_array = np.zeros((fmm.L**2)*2, dtype=ctypes.c_double)

    lsize = fmm.tree[1].parent_local_size

    eval_point = (1., 3., 4.)
    eval_sph = spherical(np.reshape(eval_point, (1, 3)))

    # print("l_array", l_array)

    point_sa[:] = np.array(point) + np.array(eval_point)
    # print("point_sa", point_sa[:])
    phi_loop.execute()

    if lsize is not None:
        moments = fmm.tree_parent[1][0, 0, 0, :]

        fmm._translate_mtl_lib['mtl_test_wrapper'](
            ctypes.c_int64(fmm.L),
            ctypes.c_double(disp_sph[0, 0]),
            extern_numpy_ptr(moments),
            extern_numpy_ptr(exp_array),
            extern_numpy_ptr(p_array),
            extern_numpy_ptr(fmm._a),
            extern_numpy_ptr(fmm._ar),
            extern_numpy_ptr(fmm._ipower_mtl),
            extern_numpy_ptr(l_array)
        )

        local_phi_re, local_phi_im = compute_phi_local(
            fmm.L, l_array, eval_sph)

        err_re = abs(local_phi_re - phi_ga[0])
        err_im = abs(local_phi_im)

        if MPIRANK == 0 and DEBUG:
            print('LOCAL ' + 60*'~')
            print("ERR RE:", err_re, "\tcomputed:", local_phi_re, "\tdirect:",
                  phi_ga[0])
            print("ERR IM:", err_im)
            print(60*'~')

        assert err_re < eps, "bad real part"
        assert err_im < 10.**-15, "bad imag part"






def test_fmm_init_5():

    E = 30.
    R = 5

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4

    fmm = PyFMM(domain=A.domain, r=R, eps=eps)

    ncubeside = 2**(fmm.R-1)
    N = ncubeside ** 3
    #N = 1
    A.npart = N


    rng = np.random.RandomState(seed=1234)
    #rng = np.random

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))

    A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))

    bias = np.sum(A.Q[:])/N

    #A.Q[:] -= bias
    #A.P[0,:] = 0.0
    #print(A.P[0,:], A.Q[0])


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

    fmm._compute_cube_contrib(A.P, A.Q)

    point = np.array([2*E,2*E,2*E])

    # compute potential energy to point across all charges directly
    src = """
    const double d0 = P.i[0] - POINT[0];
    const double d1 = P.i[1] - POINT[1];
    const double d2 = P.i[2] - POINT[2];
    phi[0] += Q.i[0] / sqrt(d0*d0 + d1*d1 + d2*d2);
    """.format(point[0], point[1], point[2])
    phi_kernel = kernel.Kernel('point_phi', src,
                               headers=(kernel.Header('math.h'),))
    phi_ga = data.GlobalArray(ncomp=1, dtype=ctypes.c_double)
    point_sa = data.ScalarArray(ncomp=3, dtype=ctypes.c_double)
    point_sa[:] = point
    phi_loop = loop.ParticleLoopOMP(kernel=phi_kernel,
                                    dat_dict={'P': A.P(access.READ),
                                              'Q': A.Q(access.READ),
                                              'POINT': point_sa(access.READ),
                                              'phi': phi_ga(access.INC_ZERO)})
    phi_loop.execute()


    for level in range(fmm.R - 1, 0, -1):
        #if MPIRANK == 0:
        #    print(level)
        phi_sph_re = 0.0
        phi_sph_im = 0.0

        lsize = fmm.tree[level].parent_local_size

        #print(MPIRANK, lsize, fmm.tree[level-1].parent_local_size)

        if lsize is not None:
            fmm._translate_m_to_m(level)
            fmm._fine_to_coarse(level)

            parent_shape = fmm.tree_plain[level][:,:,:,0].shape

            sep = A.domain.extent[0] / float(2.**(level - 1.))
            start_point = -0.5*E + 0.5*sep

            offset = fmm.tree[level].local_grid_offset

            if lsize is not None:
                for momx in itertools.product(range(parent_shape[0]//2),
                                              range(parent_shape[1]//2),
                                              range(parent_shape[2]//2)):

                    center = np.array(
                        (start_point + (offset[2]//2 + momx[2])*sep,
                        start_point + (offset[1]//2 + momx[1])*sep,
                        start_point + (offset[0]//2 + momx[0])*sep))
                    disp = point - center
                    moments = fmm.tree_parent[level][
                              momx[0], momx[1], momx[2], :]
                    disp_sph = spherical(np.reshape(disp, (1, 3)))

                    phi_sph_re1, phi_sph_im1 = compute_phi(fmm.L, moments,
                                                           disp_sph)
                    phi_sph_re += phi_sph_re1
                    phi_sph_im += phi_sph_im1

        if level < fmm.R-1:
            last_re = red_re
        else:
            last_re = 0.0

        red_re = mpi.all_reduce(np.array((phi_sph_re)))
        red_im = mpi.all_reduce(np.array((phi_sph_im)))

        red_re = abs(red_re - phi_ga[0])
        red_im = abs(red_im)

        # print(moments[:llimit**2:])
        if MPIRANK == 0 and DEBUG:
            print(60*'~')
            print("ERR RE:", red_re)
            print("ERR IM:", red_im)
            print(60*'~')

        assert red_im < 10.**-15, "bad imaginary part"
        assert red_re >= last_re, "Errors do not get better as level -> 0"
        assert red_re < eps, "error did not meet tol"

    # after traversing up tree
    disp = -1. * point #  center is (0,0,0)
    disp_sph = spherical(np.reshape(disp, (1, 3)))

    # spherical harmonics needed for point

    exp_array = np.zeros(fmm.L*8 + 2, dtype=ctypes.c_double)
    p_array = np.zeros((fmm.L*2)**2, dtype=ctypes.c_double)

    def re_lm(l,m): return (l**2) + l + m

    for lx in range(fmm.L*2):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[0,1]))

        for mxi, mx in enumerate(mrange2):
            coeff = math.sqrt(float(math.factorial(lx-abs(mx)))/
                math.factorial(lx+abs(mx)))
            p_array[re_lm(lx, mx)] = scipy_p[mxi].real*coeff

    for mxi, mx in enumerate(list(
            range(-2*fmm.L, 1)) + list(range(1, 2*fmm.L+1))
        ):

        exp_array[mxi] = np.cos(mx*disp_sph[0,2])
        exp_array[mxi + fmm.L*4 + 1] = np.sin(mx*disp_sph[0,2])

    l_array = np.zeros((fmm.L**2)*2, dtype=ctypes.c_double)

    eval_point = (0, 0., 0.0)
    eval_sph = spherical(np.reshape(eval_point, (1, 3)))

    lsize = fmm.tree[1].parent_local_size
    if lsize is not None:
        moments = fmm.tree_parent[1][0, 0, 0, :]

        fmm._translate_mtl_lib['mtl_test_wrapper'](
            ctypes.c_int64(fmm.L),
            ctypes.c_double(disp_sph[0, 0]),
            extern_numpy_ptr(moments),
            extern_numpy_ptr(exp_array),
            extern_numpy_ptr(p_array),
            extern_numpy_ptr(fmm._a),
            extern_numpy_ptr(fmm._ar),
            extern_numpy_ptr(fmm._ipower_mtl),
            extern_numpy_ptr(l_array)
        )

        local_phi_re, local_phi_im = compute_phi_local(fmm.L, l_array,
            eval_sph)

        err_re = abs(local_phi_re - phi_ga[0])
        err_im = abs(local_phi_im)

        if MPIRANK == 0 and DEBUG:
            print('LOCAL ' + 60*'~')
            print("ERR RE:", err_re, "\tcomputed:", local_phi_re, 
            "\tdirect:", phi_ga[0])
            print("ERR IM:", err_im)
            print(60*'~')

        assert err_re < eps, "bad real part"
        assert err_im < 10.**-15, "bad imag part"

    # create a second tree to traverse down.
    fmm2 = PyFMM(domain=A.domain, r=R, eps=eps)

    if MPIRANK == 0:
        fmm2.tree_parent[1][0,0,0,:] = l_array[:]

    box2_start = (-0.5*E + point[0], -0.5*E + point[1], -0.5*E + point[2])
    
    #check copied moments
    #local_phi_re, local_phi_im = compute_phi_local(
    #    fmm.L, fmm2.tree_parent[1][0,0,0,:], eval_sph)
    #print(local_phi_re, phi_ga[0])

    # on level 1 translate parent to child
    for level in range(1, fmm2.R):
        ls = fmm2.tree[level].local_grid_cube_size


        if ls is not None:
            fmm2._translate_l_to_l(level)
        if level < fmm.R-1:
            fmm2._coarse_to_fine(level)

        ncubes_s = 2**level
        cube_width = E/ncubes_s
        lo = fmm2.tree[level].local_grid_offset
        for cubex in itertools.product(range(ncubes_s), 
            range(ncubes_s), range(ncubes_s)):
            mid = [0.0, 0.0, 0.0]
            mid[0] = box2_start[0] + (0.5 + cubex[0]) * cube_width
            mid[1] = box2_start[1] + (0.5 + cubex[1]) * cube_width
            mid[2] = box2_start[2] + (0.5 + cubex[2]) * cube_width
            
            
            # execute the direct loop for this child box
            point_sa[:] = mid
            phi_loop.execute()

            #eval_sph = spherical(np.zeros((1, 3)))
            #local_phi_re, local_phi_im = compute_phi_local(
            #    fmm.L, 
            #    fmm2.tree_plain[level][cubex[2], cubex[1], cubex[0],:],
            #    eval_sph)
            
            #if this rank is the owner get phi from moments
            
            owner = fmm2.tree[level].owners[cubex[2], cubex[1], cubex[0]]
            if MPIRANK == owner:

                # zyx
                lind = (cubex[2] - lo[0], 
                        cubex[1] - lo[1],
                        cubex[0] - lo[2])

                local_phi_re = fmm2.tree_plain[level][lind[0], lind[1],
                    lind[2],0]
                local_phi_im = 0.0

                err_re = abs(phi_ga[0] - local_phi_re)
                err_im = abs(local_phi_im)

                if DEBUG:
                    print(cubex, mid)
                    print("ERR RE:", err_re, "\tcomputed:", local_phi_re, 
                    "\tdirect:", phi_ga[0])
                    print("ERR IM:", err_im)
                    print(60*'~')

                assert err_re < eps, "bad real part"
                assert err_im < 10.**-15, "bad imag part"

            MPIBARRIER()






