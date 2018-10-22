

from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator

from scipy.special import lpmv
import math
import cmath
import ctypes

c_double = ctypes.c_double

import numpy as np
from cgen import *

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

    sph_gen = SphGen(lmax)

    assign_gen = ''
    for lx in range(lmax+1):
        for mx in range(-lx, lx+1):
            assign_gen += 're_out[LMAX * {lx} + LOFFSET + {mx}] = '.format(lx=lx, mx=mx) + \
                str(sph_gen.get_y_sym(lx, mx)[0]) + ';\n'
            assign_gen += 'im_out[LMAX * {lx} + LOFFSET + {mx}] = '.format(lx=lx, mx=mx) + \
                str(sph_gen.get_y_sym(lx, mx)[1]) + ';\n'


    src = """
    #define LMAX ({LMAX})
    #define LOFFSET ({LOFFSET})

    extern "C" int test_sph_gen(
        const double theta,
        const double phi,
        double * RESTRICT re_out,
        double * RESTRICT im_out
    ){{
        {SPH_GEN}
        {ASSIGN_GEN}
        return 0;
    }}
    """.format(
        SPH_GEN=str(sph_gen.module),
        ASSIGN_GEN=str(assign_gen),
        LMAX=2*lmax+1,
        LOFFSET=lmax
    )
    header = str(sph_gen.header)

    lib = simple_lib_creator(header_code=header, src_code=src)['test_sph_gen']

    re_out = np.zeros((lmax+1, 2*lmax+1), dtype=c_double)
    im_out = np.zeros_like(re_out)

    rng = np.random.RandomState(1234)
    
    theta_set = rng.uniform(low=0.0, high=math.pi, size=10)
    phi_set = rng.uniform(low=0.0, high=2.*math.pi, size=10)

    for theta, phi in zip(theta_set, phi_set):
        lib(
            c_double(theta),
            c_double(phi),
            re_out.ctypes.get_as_parameter(),
            im_out.ctypes.get_as_parameter()
        )

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
                
                re_err = abs(scipy_real - re_out[lx, lmax + mx])
                im_err = abs(scipy_imag - im_out[lx, lmax + mx])

                assert re_err < 10.**-14
                assert im_err < 10.**-14













