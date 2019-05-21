from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

#from ppmd_vis import plot_spheres

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../../res'), filename)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb import fmm_pbc
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = False
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
    if type(xyz) is tuple or len(xyz.shape) == 1:
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

    #print("spherical", xyz, sph)
    return sph


def Afoo(n, m): return ((-1.)**n)/math.sqrt(math.factorial(n - m) * \
                                            math.factorial(n + m))
def Ifoo(k, m): return ((1.j) ** (abs(k-m) - abs(k) - abs(m)))


def Hfoo(nx, mx):
    return math.sqrt(
        float(math.factorial(nx - abs(mx)))/math.factorial(nx + abs(mx))
    )

def Pfoo(nx, mx, x):
    if abs(mx) > abs(nx):
        return 0.0
    elif nx < 0:
        return Pfoo(-1*nx -1, mx, x)
    else:
        return lpmv(mx, nx, x)

def Yfoo(nx, mx, theta, phi):
    coeff = Hfoo(nx, mx)
    legp = lpmv(abs(mx), nx, math.cos(theta))

    assert abs(legp.imag) < 10.**-16
    return coeff * legp * cmath.exp(1.j * mx * phi)



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


#@pytest.mark.skipif("True")
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
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
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

    fmm.free()


@pytest.mark.skipif("True")
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

    assert abs(bias) < 10.**-6

    local_phi_direct = 0.0

    A.scatter_data_from(0)


    t0 = time.time()
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
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

    fmm.free()


@pytest.fixture(
    scope="module",
    params=(10.**-2,10.**-4,10.**-6)
)
def tol_set(request):
    return request.param

@pytest.fixture(
    scope="module",
    params=(3,4)
)
def level_set(request):
    return request.param

@pytest.fixture(
    scope="module",
    params=(True, '27')
)
def space_set(request):
    return request.param


def test_fmm_init_5_4(level_set, tol_set, space_set):
#def test_fmm_init_5_4():

    # cannot decompose the R=2 case on more than one process
    if MPISIZE > 1 and level_set < 3:
        return

    rc = 5.
    R = level_set
    #R = 3
    eps = tol_set
    #eps = 10.**-4
    free_space = space_set
    #free_space = True


    N = 20
    #N = 50

    E = 50.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    ASYNC = False
    DIRECT = True if MPISIZE == 1 else False

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

    Q = np.sum(A.Q[:]**2.)

    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, execute_async=ASYNC)
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
    t1 = time.time()

    if DIRECT:
        phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
            if free_space == '27':
                for ofx in cube_offsets:
                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
    else:
        if free_space == '27':
            phi_direct = -0.12868996439494947981
        elif free_space == True:
            phi_direct = -0.12131955438932764957
        else:
            raise RuntimeError("bad parameter")

    local_err = abs(phi_py - phi_direct)
    if local_err/Q > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print("\n")
        #print(60*"-")
        #opt.print_profile()
        #print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("ENERGY DIRECT:\t{:.20f}".format(phi_direct))
        print("ENERGY FMM:\t", phi_py)
        print("ERR:\t\t", serr)


    assert local_err/Q < eps

    fmm.free()


def test_fmm_init_5_4_quad(level_set, tol_set, space_set):

    # cannot decompose the R=2 case on more than one process
    if MPISIZE > 1 and level_set < 3:
        return

    rc = 5.
    R = level_set
    #R = 3
    eps = tol_set
    #eps = 10.**-4
    free_space = space_set
    #free_space = True

    N = 4
    #N = 50

    E = 50.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    ASYNC = False
    DIRECT = True if MPISIZE == 1 else False

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

    ra = 0.25 * E
    nra = -0.25 * E

    A.P[0,:] = (0, nra, nra)
    A.P[1,:] = (0, nra, ra)
    A.P[2,:] = (0, ra, nra)
    A.P[3,:] = (0, ra, ra)

    A.Q[0,0] = -1.
    A.Q[3,0] = -1.
    A.Q[1,0] = 1.
    A.Q[2,0] = 1.

    Q = np.sum(A.Q[:]**2.)

    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, execute_async=ASYNC)
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
    t1 = time.time()

    if DIRECT:
        phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
            if free_space == '27':
                for ofx in cube_offsets:
                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
    else:
        if free_space == '27':
            phi_direct = -0.12823547267060314558
        elif free_space == True:
            phi_direct = -0.10343145750507620750
        else:
            raise RuntimeError("bad parameter")

    local_err = abs(phi_py - phi_direct)
    if local_err/Q > eps: serr = red(local_err)
    else: serr = green(local_err)


    if MPIRANK == 0 and DEBUG:
        print("\n")
        #print(60*"-")
        #opt.print_profile()
        #print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("ENERGY DIRECT:\t{:.20f}".format(phi_direct))
        print("ENERGY FMM:\t", phi_py)
        print("ERR:\t\t", serr)

    assert local_err/Q < eps*2


    fmm.free()


@pytest.fixture(
    scope="module",
    params=(3,4)
)
def level_set_3_4(request):
    return request.param
def test_fmm_init_5_5_nacl(level_set_3_4):
    R = level_set_3_4

    Cdata = np.load(get_res_file_path('coulomb/NACL.npy'))
    N = Cdata.shape[0]
    #E = 50.
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-6

    ASYNC = False
    EWALD = True
    free_space = False

    rc = min(10, E/4.)
    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)
    #shell_terms = fmm._test_shell_sum(26, fmm.L)
    #fmm._boundary_terms[:] = shell_terms[:]

    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            eps=10.**-14,
            shared_memory=SHARED_MEMORY
        )

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    A.P[:] = Cdata[:,0:3:]
    #print(np.min(Cdata[:,0:3:]), np.max(Cdata[:,0:3:]))

    A.P[:] -= 15.
    A.P[:] *= E/30.

    A.Q[:,0] = Cdata[:,3]

    Q = np.sum(np.abs(A.Q[:N:,0]))

    #print(A.P[:N:,:])
    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, execute_async=ASYNC)
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
    t1 = time.time()

    dipole_src = """
    dipole[0] += P.i[0] * Q.i[0];
    dipole[1] += P.i[1] * Q.i[0];
    dipole[2] += P.i[2] * Q.i[0];
    """
    dipole_ga = data.GlobalArray(ncomp=3)
    dipole_kernel = kernel.Kernel('dipole', dipole_src)
    dipole_loop = loop.ParticleLoopOMP(kernel=dipole_kernel,
                                       dat_dict={
                                           'dipole': dipole_ga(access.INC_ZERO),
                                           'P': A.P(access.READ),
                                           'Q': A.Q(access.READ)
                                       })
    dipole_loop.execute()
    if MPIRANK == 0 and DEBUG:
        print("Dipole moment:", dipole_ga[:])


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

    #print(phi_ewald, A.cri[0], A.crr[0], A.crs[0])

    local_err = abs(phi_py - phi_ewald)/Q

    assert local_err < eps

    if local_err > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print(60*"-")
        #opt.print_profile()
        print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("TIME EWALD:\t", t3 - t2)
        print("ENERGY FMM:\t", phi_py)
        print("ENERGY EWALD:\t", phi_ewald, A.cri[0], A.crr[0], A.crs[0])
        print("ERR:\t\t", serr)

    fmm.free()


# system has a dipole moment
#@pytest.mark.skipif("True")
def test_fmm_init_5_5_co2():
    R = 3

    Cdata = np.load(get_res_file_path('coulomb/CO2cuboid.npy'))
    N = Cdata.shape[0]
    E = 50.
    #E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4

    ASYNC = False
    EWALD = True
    free_space = False

    rc = min(10, E/4.)
    fmm = PyFMM(domain=A.domain, r=R, l=14, free_space=free_space)

    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            eps=10.**-14,
            shared_memory=SHARED_MEMORY
        )

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    A.P[:] = Cdata[:,0:3:]

    A.Q[:,0] = Cdata[:,3]

    #print(A.P[:N:,:])
    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, execute_async=ASYNC)
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
    t1 = time.time()

    dipole_src = """
    dipole[0] += P.i[0] * Q.i[0];
    dipole[1] += P.i[1] * Q.i[0];
    dipole[2] += P.i[2] * Q.i[0];
    """
    dipole_ga = data.GlobalArray(ncomp=3)
    dipole_kernel = kernel.Kernel('dipole', dipole_src)
    dipole_loop = loop.ParticleLoopOMP(kernel=dipole_kernel,
                                       dat_dict={
                                           'dipole': dipole_ga(access.INC_ZERO),
                                           'P': A.P(access.READ),
                                           'Q': A.Q(access.READ)
                                       })
    dipole_loop.execute()
    if MPIRANK == 0 and DEBUG:
        print("Dipole moment:", dipole_ga[:])

    phi_ewald = ewald(A.P, A.Q, A.F)
    err = abs(phi_ewald - phi_py)
    assert err < 10.**-5

    fmm.free()

@pytest.mark.skipif("MPISIZE>1")
def test_fmm_init_5_6_1():

    E = 2.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-5
    azero = 10.**-14

    fmm = PyFMM(domain=A.domain, r=3, eps=eps, free_space=False)

    #shell_terms = np.load(get_res_file_path('coulomb/r_coeffs_e2_L32.npy'))


    shelltest2 = fmm_pbc._shell_test_2_FMMPbc(
        fmm.L, fmm.eps,fmm.domain, fmm.dtype)
    shell_terms = shelltest2._test_shell_sum2(16, fmm.L)

    for nx in range(fmm.L*2):
        for mx in range(-1*nx, nx+1):
            ev = fmm._boundary_terms[fmm.re_lm(nx, mx)]
            dv = shell_terms[fmm.re_lm(nx, mx)]

            err = abs(ev - dv)
            if DEBUG:
                if err > eps: serr = red(err)
                else: serr = green(err)
                print(nx, mx, serr, ev, dv)

            assert err < eps, "{} {}".format(ev, dv)

            if (nx % 2) == 1 or (mx % 2) == 1:
                assert abs(ev) < azero
            if (mx % 4) != 0:
                assert abs(ev) < azero
            #if (mx < 0):
            #    assert abs(ev) < azero
            if (nx == 2):
                assert abs(ev) < azero

    #np.save('/tmp/r_coeffs.npy', shell_terms)

    rterms = fmm._boundary_terms[:]

    rng = np.random.RandomState(seed=1234)
    testo = rng.uniform(low=-1., high=.1, size=(fmm.L**2)*2)

    fmm.tree_parent[1][0, 0, 0, :] = testo[:]
    fmm._compute_periodic_boundary()

    def Afoo(n, m): return ((-1.)**n)/math.sqrt(math.factorial(n - m) * \
                                                math.factorial(n + m))
    def Ifoo(k, m): return ((1.j) ** (abs(k-m) - abs(k) - abs(m))).real

    l00 = fmm.tree_parent[1][0, 0, 0, :]
    for jx in range(fmm.L):
        for kx in range(-1*jx, jx+1):
            contrib_re = 0.0
            contrib_im = 0.0
            for nx in range(fmm.L):
                for mx in range(-1*nx, nx+1):

                    coeff = Ifoo(kx, mx) * Afoo(nx, mx) * Afoo(jx, kx) / \
                        Afoo(jx + nx, mx - kx) * ((-1.)**nx) * \
                        rterms[fmm.re_lm(jx+nx, mx-kx)]

                    contrib_re += testo[fmm.re_lm(nx, mx)] * coeff
                    contrib_im += testo[fmm.im_lm(nx, mx)] * coeff

            assert abs(contrib_re - l00[fmm.re_lm(jx, kx)]) < 10.**-15
            assert abs(contrib_im - l00[fmm.im_lm(jx, kx)]) < 10.**-15


    fmm.free()

@pytest.mark.skipif("MPISIZE>1")
def test_fmm_init_5_6_2():

    E = 50.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-5
    azero = 10.**-9
    fmm = PyFMM(domain=A.domain, r=3, eps=eps, free_space=False)


    shelltest2 = fmm_pbc._shell_test_2_FMMPbc(
        fmm.L, fmm.eps,fmm.domain, fmm.dtype)
    shell_terms = shelltest2._test_shell_sum(30, fmm.L)

    for nx in range(fmm.L*2):
        for mx in range(-1*nx, nx+1):
            ev = fmm._boundary_terms[fmm.re_lm(nx, mx)]
            dv = shell_terms[fmm.re_lm(nx, mx)]

            err = abs(ev - dv)
            if DEBUG:
                if err > eps: serr = red(err)
                else: serr = green(err)
                print(nx, mx, serr)

            assert err < eps

            if (nx % 2) == 1 or (mx % 2) == 1:
                assert abs(ev) < azero
            if (mx % 4) != 0:
                assert abs(ev) < azero
            if (mx < 0):
                assert abs(ev) < azero
            if (nx == 2):
                assert abs(ev) < azero

    fmm.free()


def test_fmm_init_5_4_quad2():

    R = 3
    eps = 10.**-5
    free_space = True

    N = 8
    E = 4.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    ASYNC = False
    DIRECT = True if MPISIZE == 1 else False

    DIRECT= True

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

    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        #A.P[0,:] = (0, nra, nra)
        #A.P[1,:] = (0, nra, ra)
        #A.P[2,:] = (0, ra, nra)
        #A.P[3,:] = (0, ra, ra)

        A.P[0,:] = (nra, nra, 0.1)
        A.P[1,:] = (nra, ra, 0.1)
        A.P[2,:] = (ra, nra, 0.1)
        A.P[3,:] = (ra, ra, 0.1)

        A.Q[0,0] = -1.
        A.Q[3,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = 1.

    elif N == 2:
        A.P[0,:] = (-0.25*E-0.1, 0, 0)
        A.P[1,:] = (0.25*E-0.1, 0, 0)

        #A.P[0,:] = (0, -0.25*E, 0)
        #A.P[1,:] = (0, 0.25*E, 0)
        #A.P[0,:] = (0, 0, -0.25*E)
        #A.P[1,:] = (0, 0, 0.25*E)

        A.Q[:,0] = 1.
        A.Q[0,0] = -1.

    elif N == 8:
        for px in range(8):
            phi = (float(px)/8) * 2. * math.pi
            pxr = 0.25*E
            pxx = pxr * math.cos(phi)
            pxy = pxr * math.sin(phi)

            A.P[px, :] = (0,pxx, pxy)
            A.Q[px, 0] = 1. - 2. * (px % 2)


    A.scatter_data_from(0)

    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, execute_async=ASYNC)
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
    t1 = time.time()

    if DIRECT:
        phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
            if free_space == '27':
                for ofx in cube_offsets:
                    cube_mid = np.array(ofx)*E
                    for jx in range(N):
                        rij = np.linalg.norm(A.P[jx,:] + cube_mid - A.P[ix, :])
                        phi_direct += 0.5*A.Q[ix, 0] * A.Q[jx, 0] /rij
    else:
        if free_space == '27':
            phi_direct = -0.12868996439494947981
        elif free_space == True:
            phi_direct = -0.12131955438932764957
        else:
            raise RuntimeError("bad parameter")

    local_err = abs(phi_py - phi_direct)
    if local_err > eps: serr = red(local_err)
    else: serr = green(local_err)

    if MPIRANK == 0 and DEBUG:
        print("\n")
        #print(60*"-")
        #opt.print_profile()
        #print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("ENERGY DIRECT:\t{:.20f}".format(phi_direct))
        print("ENERGY FMM:\t", phi_py)
        print("ERR:\t\t", serr)

    #assert local_err < eps

    fmm.free()


@pytest.mark.skipif("MPISIZE>1")
def test_fmm_init_5_7_quad():
    R = 3

    N = 2
    E = 4.

    rc = E/2


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-4

    ASYNC = False
    EWALD = True
    free_space = False

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)


    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            eps=10.**-12,
            shared_memory=SHARED_MEMORY
        )

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    if N == 2:
        A.P[0,:] = (-0.25*E, 0.1, 0.1)
        A.P[1,:] = (0.25*E, 0.1, 0.1)

        A.Q[:,0] = 1.
        A.Q[0,0] = -1
    elif N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        #A.P[0,:] = (nra, nra, 0)
        #A.P[1,:] = (nra, ra, 0)
        #A.P[2,:] = (ra, nra, 0)
        #A.P[3,:] = (ra, ra, 0)

        A.P[0,:] = (0.0, nra, nra)
        A.P[1,:] = (0.0, nra, ra)
        A.P[2,:] = (0.0, ra, nra)
        A.P[3,:] = (0.0, ra, ra)

        A.Q[0,0] = -1.
        A.Q[3,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = 1.



    A.scatter_data_from(0)


    t0 = time.time()
    #phi_py = fmm._test_call(A.P, A.Q, execute_async=ASYNC)
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)

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
        #opt.print_profile()
        print(60*"-")
        print("TIME FMM:\t", t1 - t0)
        print("TIME EWALD:\t", t3 - t2)
        print("ENERGY FMM:\t", phi_py)
        print("ENERGY EWALD:\t", phi_ewald, A.cri[0], A.crr[0], A.crs[0])
        print("ERR:\t\t", serr)

    if N == 4:
        dl_phi = -2.324071E+01
    elif N == 2:
        dl_phi = -9.868676E+00

    print("DLPOLY EWALD (EV):\t", dl_phi)
    print("EV EWALD:\t", ewald.internal_to_ev()*phi_ewald, "\t err:\t",
          abs(dl_phi - ewald.internal_to_ev()*phi_ewald))
    print("EV FMM:  \t", ewald.internal_to_ev()*phi_py, "\t err:\t",
          abs(dl_phi - ewald.internal_to_ev()*phi_py))

    fmm.free()


def nacl_lattice(crn, e, sd=0.05, seed=87712846):

    raw_lattice = utility.lattice.cubic_lattice(crn, e)

    N = crn[0]*crn[1]*crn[2]

    charges = np.zeros(N)
    raw_labels = ('Na', 'Cl')
    labels = np.zeros(N, dtype='|S2')

    raw_charges = (1.0, -1.0)
    counts = [0, 0]
    lattice = np.zeros((N,3))

    nai = 0
    cli = -1

    for px in range(N):
        ix = px % crn[0]
        iy = ((px - ix) // crn[0]) % crn[1]
        iz = (px - ix - iy*crn[0])//(crn[0]*crn[1])

        t = (ix + iy + iz) % 2

        if t == 0:
            idx = nai
            nai += 1
        else:
            idx = cli
            cli -= 1

        charges[idx] = raw_charges[t]
        lattice[idx, :] = raw_lattice[px, :]
        labels[idx] = raw_labels[t]
        counts[t] += 1

    rng = np.random.RandomState(seed=seed)
    velocities = rng.normal(0.0, sd, size=(N,3))

    return lattice, velocities, charges, labels, counts


def test_fmm_init_5_7_cube():
    R = 3

    crN = 20
    N = crN**3

    E = 3.*crN

    rc = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-6

    ASYNC = False
    EWALD = True
    free_space = False

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space)


    if EWALD:
        ewald = EwaldOrthoganalHalf(
            domain=A.domain,
            real_cutoff=rc,
            eps=eps,
            shared_memory=SHARED_MEMORY
        )

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)

    lattice, velocities, charges, labels, counts = nacl_lattice(
        (crN, crN, crN), (E, E, E))


    A.P[:] = lattice[:]
    A.Q[:,0] = charges[:]

    bias = np.sum(A.Q[:])/N
    A.Q[:] -= bias

    if MPIRANK == 0 and DEBUG:
        print("charge sum:", np.sum(A.Q[:N:,0]))

    A.scatter_data_from(0)

    dipole_src = """
    dipole[0] += P.i[0] * Q.i[0];
    dipole[1] += P.i[1] * Q.i[0];
    dipole[2] += P.i[2] * Q.i[0];
    """
    dipole_ga = data.GlobalArray(ncomp=3)
    dipole_kernel = kernel.Kernel('dipole', dipole_src)
    dipole_loop = loop.ParticleLoopOMP(kernel=dipole_kernel,
                                       dat_dict={
                                           'dipole': dipole_ga(access.INC_ZERO),
                                           'P': A.P(access.READ),
                                           'Q': A.Q(access.READ)
                                       })
    dipole_loop.execute()



    t0 = time.time()
    phi_py = fmm(A.P, A.Q, execute_async=ASYNC)
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

    if MPIRANK == 0 and DEBUG:
        print("Dipole moment:", dipole_ga[:])

    fmm.free()


