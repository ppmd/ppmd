

from ppmd.coulomb.legendre import ALegendrePolynomialGen
from ppmd.lib.build import simple_lib_creator

from scipy.special import lpmv
import math
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
                lpmv_gen.get_p_sym(lx, mx) + ';\n'

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


    for theta in np.random.uniform(low=-0.999999, high=0.999999, size=10):
        lib(c_double(theta), lpmv_c.ctypes.get_as_parameter())
        for lx in range(lmax+1):
            for mx in range(lx+1):
                correct = lpmv(mx, lx, theta)
                rel = 1 if (abs(correct) < 1.0) else abs(correct)
                err = abs(correct - lpmv_c[lx, mx]) / rel
                assert err < 10.**-12










