

from ppmd.coulomb.sph_harm import *
from ppmd.coulomb.fmm_pbc import *
from ppmd.lib.build import simple_lib_creator

from scipy.special import lpmv
import math
import cmath
import ctypes

c_double = ctypes.c_double

import numpy as np
from cgen import *
from itertools import product

def test_legendre_gen_1():

    lmax = 24

    lpmv_gen = ALegendrePolynomialGen(lmax)
    
    assign_gen = ''
    for lx in range(lmax+1):
        for mx in range(lx+1):
            assign_gen += 'out[LMAX * {lx} + {mx}] = '.format(lx=lx, mx=mx) + \
                str(lpmv_gen.get_p_sym(lx, mx)) + ';\n'

    src = """
    #define LMAX ({LMAX})
    extern "C" int test_lpmv_gen(
        const double theta,
        double * RESTRICT out
    ){{
        {LPMV_GEN}
        {ASSIGN_GEN}
        return 0;
    }}
    """.format(
        LPMV_GEN=str(lpmv_gen.module),
        ASSIGN_GEN=str(assign_gen),
        LMAX=lmax+1
    )
    header = str(lpmv_gen.header)

    lib = simple_lib_creator(header_code=header, src_code=src)['test_lpmv_gen']
    
    lpmv_c = np.zeros((lmax+1, lmax+1), dtype=c_double)

    rng = np.random.RandomState(14523)
    for theta in rng.uniform(low=-0.999999, high=0.999999, size=10):
        lib(c_double(theta), lpmv_c.ctypes.get_as_parameter())
        for lx in range(lmax+1):
            for mx in range(lx+1):
                correct = lpmv(mx, lx, theta)
                rel = 1 if (abs(correct) < 1.0) else abs(correct)
                err = abs(correct - lpmv_c[lx, mx]) / rel
                assert err < 10.**-12


def test_exp_gen_1():
    lmax = 40

    exp_gen = SphExpGen(lmax)

    assign_gen = ''
    for lx in range(-lmax, lmax+1):
        assign_gen += 're_out[{lx}] = {exp};\n'.format(lx=lx, exp=exp_gen.get_e_sym(lx)[0])
        assign_gen += 'im_out[{lx}] = {exp};\n'.format(lx=lx, exp=exp_gen.get_e_sym(lx)[1])
    
    src = """
    extern "C" int test_exp_gen(
        const double phi,
        double * RESTRICT re_out,
        double * RESTRICT im_out
    ){{
        {EXP_GEN}
        {ASSIGN_GEN}
        return 0;
    }}
    """.format(
        EXP_GEN=str(exp_gen.module),
        ASSIGN_GEN=str(assign_gen)
    )
    header = str(exp_gen.header)

    lib = simple_lib_creator(header_code=header, src_code=src)['test_exp_gen']

    re_exp_c = np.zeros(2*lmax+1, dtype=c_double)
    im_exp_c = np.zeros_like(re_exp_c)
    
    rng = np.random.RandomState(1452)
    for phi in rng.uniform(low=0.0, high=2.*math.pi, size=20):
        lib(c_double(phi),
            re_exp_c[lmax:].view().ctypes.get_as_parameter(), 
            im_exp_c[lmax:].view().ctypes.get_as_parameter()
        )
        for lx in range(-lmax, lmax+1):
            correct = cmath.exp(lx*phi*1.j)
            re_err = abs(re_exp_c[lmax + lx] - correct.real)
            im_err = abs(im_exp_c[lmax + lx] - correct.imag)

            assert re_err < 10.**-13
            assert im_err < 10.**-13


def test_sph_gen_1():

    lmax = 26
    N = 10
    M = (lmax+1) * (2*lmax+1)

    sph_gen = SphGen(lmax)

    assign_gen = ''
    for lx in range(lmax+1):
        for mx in range(-lx, lx+1):
            assign_gen += 're_out[NSTRIDE * ix + LMAX * {lx} + LOFFSET + {mx}] = '.format(lx=lx, mx=mx) + \
                str(sph_gen.get_y_sym(lx, mx)[0]) + ';\n'
            assign_gen += 'im_out[NSTRIDE * ix + LMAX * {lx} + LOFFSET + {mx}] = '.format(lx=lx, mx=mx) + \
                str(sph_gen.get_y_sym(lx, mx)[1]) + ';\n'


    src = """
    #define LMAX ({LMAX})
    #define LOFFSET ({LOFFSET})
    #define N ({N})
    #define NSTRIDE ({NSTRIDE})

    extern "C" int test_sph_gen(
        const double * RESTRICT theta_set,
        const double * RESTRICT phi_set,
        double * RESTRICT re_out,
        double * RESTRICT im_out
    ){{
        #pragma omp parallel for
        for (int ix=0; ix<N ; ix++){{
            const double theta = theta_set[ix];
            const double phi = phi_set[ix];
        {SPH_GEN}
        {ASSIGN_GEN}
        }}
        return 0;
    }}
    """.format(
        SPH_GEN=str(sph_gen.module),
        ASSIGN_GEN=str(assign_gen),
        LMAX=2*lmax+1,
        LOFFSET=lmax,
        N=N,
        NSTRIDE=M
    )
    header = str(sph_gen.header)

    lib = simple_lib_creator(header_code=header, src_code=src)['test_sph_gen']
    

    re_out = np.zeros((N, lmax+1, 2*lmax+1), dtype=c_double)
    im_out = np.zeros_like(re_out)

    rng = np.random.RandomState(1234)
    
    theta_set = np.array(rng.uniform(low=0.0, high=math.pi, size=N), dtype=c_double)
    phi_set = np.array(rng.uniform(low=0.0, high=2.*math.pi, size=N), dtype=c_double)
    lib(
        theta_set.ctypes.get_as_parameter(),
        phi_set.ctypes.get_as_parameter(),
        re_out.ctypes.get_as_parameter(),
        im_out.ctypes.get_as_parameter()
    )
    
    for ix in range(N):
        theta = theta_set[ix]
        phi = phi_set[ix]

        for lx in range(lmax + 1):
            mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
            mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
            scipy_p = lpmv(mrange, lx, np.cos(theta))

            for mxi, mx in enumerate(mrange2):

                re_exp = math.cos(mx * phi)
                im_exp = math.sin(mx * phi)

                val = math.sqrt(math.factorial(
                    lx - abs(mx))/math.factorial(lx + abs(mx)))
                val *= scipy_p[mxi]

                scipy_real = re_exp * val
                scipy_imag = im_exp * val
                
                re_err = abs(scipy_real - re_out[ix, lx, lmax + mx])
                im_err = abs(scipy_imag - im_out[ix, lx, lmax + mx])

                assert re_err < 10.**-13
                assert im_err < 10.**-13



def test_sph_pbc_gen_1():
    lmax = 20
    im_offset = (lmax + 1) ** 2
    N = 127
    gen = SphShellSum(lmax)

    rng = np.random.RandomState(1234)
    
    radius_set = np.array(rng.uniform(low=1.0, high=2.0, size=N), dtype=c_double)
    theta_set = np.array(rng.uniform(low=0.0, high=math.pi, size=N), dtype=c_double)
    phi_set = np.array(rng.uniform(low=0.0, high=2.*math.pi, size=N), dtype=c_double)
    out = np.zeros(gen.ncomp, dtype=c_double)

    gen(radius_set, theta_set, phi_set, out)

    def reY(L, M): return ((L) * ( (L) + 1 ) + (M))
    def imY(L, M): return ((L) * ( (L) + 1 ) + (M) + im_offset)
    
    correct_out = np.zeros_like(out)

    for ix in range(N):
        radius = radius_set[ix]
        theta = theta_set[ix]
        phi = phi_set[ix]

        for lx in range(lmax + 1):
            mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
            mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
            scipy_p = lpmv(mrange, lx, np.cos(theta))

            for mxi, mx in enumerate(mrange2):

                re_exp = math.cos(mx * phi)
                im_exp = math.sin(mx * phi)

                val = math.sqrt(math.factorial(
                    lx - abs(mx))/math.factorial(lx + abs(mx)))
                val *= scipy_p[mxi]

                scipy_real = re_exp * val / (radius ** (lx + 1))
                scipy_imag = im_exp * val / (radius ** (lx + 1))
                
                correct_out[reY(lx, mx)] += scipy_real
                correct_out[imY(lx, mx)] += scipy_imag
    
    for tx in range(gen.ncomp):
        err = abs(correct_out[tx] - out[tx])
        assert err < 10.**-13





def test_sph_pbc_gen_2():
    lmax = 2
    lold = (lmax + 1)

    pbc_lold = (2*lold)
    pbc_lmax = pbc_lold - 1

    
    E = 1.0
    im_offset = pbc_lold ** 2

    rc = 16.000001

    def reY(L, M): return ((L) * ( (L) + 1 ) + (M))
    def imY(L, M): return ((L) * ( (L) + 1 ) + (M) + im_offset)

    def cart_to_sph(xyz):
        dx = xyz[0]; dy = xyz[1]; dz = xyz[2]

        dx2dy2 = dx*dx + dy*dy
        r2 = dx2dy2 + dz*dz
        if r2 > (rc * rc): return None

        radius = math.sqrt(r2)
        phi = math.atan2(dy, dx)
        theta = math.atan2(math.sqrt(dx2dy2), dz)

        return radius, theta, phi

    gen = SphShellSum(pbc_lmax)

    class DummyDomain:
        def __init__(self, e):
            self.extent = np.array((e,e,e), dtype=c_double)

    domain = DummyDomain(E)

    orig_pbc = FMMPbc(lold, 10.**-15, domain, c_double)
    orig_vals = orig_pbc.compute_f() + orig_pbc.compute_g()


    dmax = int(rc/E)
    drange = list(range(-dmax, dmax+1))

    radius_list = []
    theta_list = []
    phi_list = []

    for imagex in product(drange, drange, drange):
        nrm = max(abs(imagex[0]), abs(imagex[1]), abs(imagex[2]))
        if nrm < 2:
            continue

        sph_coord = cart_to_sph(imagex)

        if sph_coord is not None:
            radius_list.append(sph_coord[0])
            theta_list.append(sph_coord[1])
            phi_list.append(sph_coord[2])


    radius_set = np.array(radius_list, dtype=c_double)
    theta_set = np.array(theta_list, dtype=c_double)
    phi_set = np.array(phi_list, dtype=c_double)
    out = np.zeros(gen.ncomp, dtype=c_double)

    #print("\n" + "-" * 80)
    #print(orig_vals)
    #print("-" * 80)

    gen(radius_set, theta_set, phi_set, out)
    
    #print(out)
    #print("-" * 80)


    N = len(phi_set)
    def reY(L, M): return ((L) * ( (L) + 1 ) + (M))
    def imY(L, M): return ((L) * ( (L) + 1 ) + (M) + im_offset)
    
    correct_out = np.zeros_like(out)

    for ix in range(N):
        radius = radius_set[ix]
        theta = theta_set[ix]
        phi = phi_set[ix]

        for lx in range(pbc_lold):
            mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
            mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
            scipy_p = lpmv(mrange, lx, np.cos(theta))

            for mxi, mx in enumerate(mrange2):

                re_exp = math.cos(mx * phi)
                im_exp = math.sin(mx * phi)

                val = math.sqrt(math.factorial(
                    lx - abs(mx))/math.factorial(lx + abs(mx)))
                val *= scipy_p[mxi]

                scipy_real = re_exp * val / (radius ** (lx + 1))
                scipy_imag = im_exp * val / (radius ** (lx + 1))
                
                correct_out[reY(lx, mx)] += scipy_real
                correct_out[imY(lx, mx)] += scipy_imag
    
    for lx in range(lold):
        for mx in range(-lx, lx+1):
            re_err = abs(correct_out[reY(lx, mx)] - out[reY(lx, mx)])
            im_err = abs(correct_out[imY(lx, mx)] - out[imY(lx, mx)])
            #print(lx, mx, "|", reY(lx, mx), re_err, "|", imY(lx, mx), im_err)
            assert re_err < 10.**-10
            assert im_err < 10.**-10


    #import ppmd.coulomb.fmm_pbc
    #pyshell = ppmd.coulomb.fmm_pbc._shell_test_2_FMMPbc(lold, None, domain, c_double)
    #shell_out = pyshell._test_shell_sum2(dmax)
    #print(shell_out)






