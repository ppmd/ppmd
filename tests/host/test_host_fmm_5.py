from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)
#from ppmd_vis import plot_spheres

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../res'), filename)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

def red(input):
    try:
        from termcolor import colored
        return colored(input, 'red')
    except Exception as e: return input
def green(input):
    try:
        from termcolor import colored
        return colored(input, 'green')
    except Exception as e: return input
def yellow(input):
    try:
        from termcolor import colored
        return colored(input, 'yellow')
    except Exception as e: return input

cube_offsets = (
    (-1,1,-1),
    (-1,-1,-1),
    (-1,0,-1),
    (0,1,-1),
    (0,-1,-1),
    (0,0,-1),
    (1,0,-1),
    (1,1,-1),
    (1,-1,-1),

    (-1,1,0),
    (-1,0,0),
    (-1,-1,0),
    (0,-1,0),
    (0,1,0),
    (1,0,0),
    (1,1,0),
    (1,-1,0),

    (-1,0,1),
    (-1,1,1),
    (-1,-1,1),
    (0,0,1),
    (0,1,1),
    (0,-1,1),
    (1,0,1),
    (1,1,1),
    (1,-1,1)
)

def tuple_it(*args, **kwargs):
    if len(kwargs) == 0:
        tx = args[0]
        return itertools.product(range(tx[0]), range(tx[1]), range(tx[2]))
    else:
        l = kwargs['low']
        h = kwargs['high']
        return itertools.product(range(l[0], h[0]),
                                 range(l[1], h[1]),
                                 range(l[2], h[2]))


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


def get_p_exp(fmm, disp_sph):
    def re_lm(l,m): return (l**2) + l + m
    exp_array = np.zeros(fmm.L*8 + 2, dtype=ctypes.c_double)
    p_array = np.zeros((fmm.L*2)**2, dtype=ctypes.c_double)
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

    return p_array, exp_array


def test_fmm_init_5_1():
    rc = 10.
    R = 3
    Ns = 10
    E = max(3.*Ns, 3*rc)

    ASYNC = False
    EWALD = True

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4
    eps2 = 10.**-3

    N = Ns**3
    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=False)
    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            shared_memory=SHARED_MEMORY
        )

    print(fmm.R, fmm.L)

    #N = 2
    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)


    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)


    A.P[:] = utility.lattice.cubic_lattice((Ns, Ns, Ns),
                                           (E, E, E))
    #A.P[:] = rng.uniform(low=-0.499*E, high=0.499*E, size=(N,3))
    A.Q[:] = rng.uniform(low=-1.0, high=1.0, size=(N,1))

    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias

    local_phi_direct = 0.0

    A.scatter_data_from(0)


    t0 = time.time()
    phi_py = fmm(A.P, A.Q, async=ASYNC)
    t1 = time.time()


    t2 = time.time()
    if EWALD:
        ewald.evaluate_contributions(positions=A.P, charges=A.Q)
        A.cri[0] = 0.0
        ewald.extract_forces_energy_reciprocal(A.P, A.Q, A.F, A.cri)
        A.crr[0] = 0.0
        ewald.extract_forces_energy_real(A.P, A.Q, A.F, A.crr)
        A.crs[0] = 0.0
        ewald.evaluate_self_interactions(A.Q, A.crs)
    t3 = time.time()

    phi_ewald = A.cri[0] + A.crr[0] + A.crs[0]

    local_err = abs(phi_py - local_phi_direct)

    if local_err > eps: serr = red(local_err)
    else: serr = yellow(local_err)

    if MPIRANK == 0 and DEBUG:
        print(60*"-")
        opt.print_profile()
        print(60*"-")
        print("TIME FMM:", t1 - t0)
        print("TIME EWALD:", t3 - t2)
        print("ENERGY FMM:", phi_py)
        print("ENERGY EWALD:", phi_ewald, A.cri[0], A.crr[0], A.crs[0])
        #print("LOCAL PHI ERR:", serr, phi_py, green(local_phi_direct))





def test_fmm_init_5_2():
    rc = 10.
    R = 3
    Ns = 2
    E = max(3.*Ns, 3*rc)

    ASYNC = False
    EWALD = True

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(1.,1.,1.))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=False)

    for lx in range(2, fmm.L, 2):
        print(lx, "parameters", fmm._compute_parameters(lx))


def test_fmm_init_5_3():
    rc = 1.
    R = 2

    N = 9
    E = 4.

    ASYNC = False
    EWALD = False

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-6

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=False)
    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            shared_memory=SHARED_MEMORY
        )

    print(fmm.R, fmm.L)

    #N = 2
    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    A.P[0:8:,  :] = utility.lattice.cubic_lattice((2, 2, 2),
                                                  (E, E, E))
    A.P[8, :] = (0.,0.,0.)

    A.Q[0:8:, :] = 1.0
    A.Q[8, 0] = -8.0

    bias = np.sum(A.Q[:])/N

    print(A.P[:])

    assert abs(bias) < 10.**-6

    local_phi_direct = 0.0

    A.scatter_data_from(0)


    t0 = time.time()
    phi_py = fmm(A.P, A.Q, async=ASYNC)
    t1 = time.time()


    t2 = time.time()
    if EWALD:
        ewald.evaluate_contributions(positions=A.P, charges=A.Q)
        A.cri[0] = 0.0
        ewald.extract_forces_energy_reciprocal(A.P, A.Q, A.F, A.cri)
        A.crr[0] = 0.0
        ewald.extract_forces_energy_real(A.P, A.Q, A.F, A.crr)
        A.crs[0] = 0.0
        ewald.evaluate_self_interactions(A.Q, A.crs)
    t3 = time.time()

    phi_ewald = A.cri[0] + A.crr[0] + A.crs[0]

    local_err = abs(phi_py - phi_ewald)

    if local_err > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print(60*"-")
        opt.print_profile()
        print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("TIME EWALD:\t", t3 - t2)
        print("ENERGY FMM:\t", phi_py)
        print("ENERGY EWALD:\t", phi_ewald, A.cri[0], A.crr[0], A.crs[0])
        print("ERR:\t\t", serr)


@pytest.fixture(
    scope="module",
    params=(10.**-2,10.**-4,10.**-6)
)
def tol_set(request):
    return request.param

@pytest.fixture(
    scope="module",
    params=(2,3,4)
)
def level_set(request):
    return request.param

#def test_fmm_init_5_4(level_set, tol_set):
def test_fmm_init_5_4():

    # cannot decompose the R=2 case on more than one process
    #if MPISIZE > 1 and level_set < 3:
    #    return

    rc = 5.
    R = level_set
    R = 3
    eps = 10.**-2
    #eps = tol_set


    N = 20
    E = 50.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()



    ASYNC = False
    DIRECT = True if MPISIZE == 1 else False
    free_space = False

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=1234)
    A.P[:] = rng.uniform(low=-0.4999*E, high=0.49999*E, size=(N,3))
    A.Q[:] = rng.uniform(low=-1., high=1., size=(N,1))
    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias

    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, async=ASYNC)
    phi_py = fmm(A.P, A.Q, async=ASYNC)
    t1 = time.time()

    if DIRECT:
        phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
            if free_space == False:
                for ofx in cube_offsets:
                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
    else:
        phi_direct = -0.12867248123756441780

    local_err = abs(phi_py - phi_direct)
    if local_err > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print(60*"-")
        #opt.print_profile()
        print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("ENERGY DIRECT:\t{:.20f}".format(phi_direct))
        print("ENERGY FMM:\t", phi_py)
        print("ERR:\t\t", serr)

    assert local_err < eps

def test_fmm_init_5_5():
    rc = 5.
    R = 3

    Cdata = np.load(get_res_file_path('coulomb/CO2cuboid.npy'))
    N = Cdata.shape[0]
    #N = 10
    E = 50.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-3

    ASYNC = False
    EWALD = True
    DIRECT = False
    free_space = False

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)
    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            shared_memory=SHARED_MEMORY
        )

    print(fmm.R, fmm.L)

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    if N == Cdata.shape[0]:
        A.P[:] = Cdata[:,0:3:]
        A.Q[:,0] = Cdata[:,3]
    elif N == 1:
        A.P[0,:] = (0.,0.,0.)
        A.Q[0,0] = 1.0
    elif N == 2:
        A.P[0,:] = (-0.25*E, 0, 0)
        A.P[1,:] = (0.25*E, 0, 0)
        A.Q[:] = 1.0
        print(A.P[0:N:,:])
    else:
        rng = np.random.RandomState(seed=1234)
        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        A.Q[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,1))
        bias = np.sum(A.Q[:])/N
        A.Q[:] -= bias

    A.scatter_data_from(0)


    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, async=ASYNC)
    phi_py = fmm._test_call(A.P, A.Q, async=ASYNC)
    t1 = time.time()


    t2 = time.time()
    if EWALD:
        ewald.evaluate_contributions(positions=A.P, charges=A.Q)
        A.cri[0] = 0.0
        ewald.extract_forces_energy_reciprocal(A.P, A.Q, A.F, A.cri)
        A.crr[0] = 0.0
        ewald.extract_forces_energy_real(A.P, A.Q, A.F, A.crr)
        A.crs[0] = 0.0
        ewald.evaluate_self_interactions(A.Q, A.crs)
    t3 = time.time()

    phi_ewald = A.cri[0] + A.crr[0] + A.crs[0]


    if DIRECT:
        phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
            if free_space == False:
                for ofx in cube_offsets:
                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
    else:
        phi_direct = -422.572495074

    local_err = abs(phi_py - phi_direct)
    if local_err > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print(60*"-")
        #opt.print_profile()
        print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("TIME EWALD:\t", t3 - t2)
        print("ENERGY DIRECT:\t", phi_direct)
        print("ENERGY FMM:\t", phi_py)
        print("ENERGY EWALD:\t", phi_ewald, A.cri[0], A.crr[0], A.crs[0])
        print("ERR:\t\t", serr)

    print(fmm.tree_halo[1][:,:,:,0])
    print(fmm.tree_plain[1][:,:,:,0])








def test_fmm_init_5_6():
    rc = 5.
    R = 3

    Cdata = np.load(get_res_file_path('coulomb/CO2cuboid.npy'))
    N = Cdata.shape[0]
    #N = 10
    E = 50.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-3

    ASYNC = False
    EWALD = True
    DIRECT = False
    free_space = False

    #fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)
    ewald = EwaldOrthoganal(
        domain=A.domain,
        real_cutoff=rc,
        shared_memory=SHARED_MEMORY
    )

    #print(fmm.R, fmm.L)



