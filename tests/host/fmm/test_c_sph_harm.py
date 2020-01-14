from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np



from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from ppmd.lib.build import simple_lib_creator
from scipy.special import sph_harm, lpmv
import time

from math import *
from itertools import product
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double

from ppmd.coulomb.sph_harm import LocalExpEval, MultipoleExpCreator, MultipoleDotVecCreator, SphGen, SphGenEphemeral, py_local_exp


def test_c_sph_harm_1():
    rng = np.random.RandomState(9476213)
    N = 20
    L = 20
    ncomp = (L**2)*2

    lee = LocalExpEval(L-1)
    
    for tx in range(20):
        point = [0, 0, 0]
        point[0] = rng.uniform(0, 10)
        point[1] = rng.uniform(0, 2*pi)
        point[2] = rng.uniform(0, pi)

        moments = rng.uniform(size=ncomp)
        rec = lee(moments, point)
        rep = lee.py_compute_phi_local(moments, point)
        rel = abs(rep)
        assert abs(rec - rep) / rel < 10.**-12

    for tx in range(20):
        point = [0.5,         1.57079633, -1.57079633]
        point[0] = rng.uniform(0, 10)
        point[1] = rng.uniform(0, 2*pi)
        point[2] = rng.uniform(0, pi)

        moments = rng.uniform(size=ncomp)
        rec = lee(moments, point)
        rep = lee.py_compute_phi_local(moments, point)
        rel = abs(rep)
        assert abs(rec - rep) / rel < 10.**-12


    for tx in range(20):
        point = [0.5,         1.57079633, 1.57079633]
        point[0] = rng.uniform(0, 10)
        point[1] = rng.uniform(0, 2*pi)
        point[2] = rng.uniform(0, pi)

        moments = rng.uniform(size=ncomp)
        rec = lee(moments, point)
        rep = lee.py_compute_phi_local(moments, point)
        rel = abs(rep)
        assert abs(rec - rep) / rel < 10.**-12





def test_c_sph_harm_2():
    rng = np.random.RandomState(9473)
    N = 20
    L = 20
    ncomp = (L**2)*2

    lee = MultipoleExpCreator(L-1)
    lee2 = MultipoleDotVecCreator(L-1)
    
    for tx in range(20):

        radius = rng.uniform()
        phi = rng.uniform(0, math.pi * 2)
        theta = rng.uniform(0, math.pi)

        sph = (radius, theta, phi)

        correct = np.zeros(ncomp, REAL)
        correctd = np.zeros(ncomp, REAL)
        to_test = np.zeros(ncomp, REAL)
        to_testm = np.zeros(ncomp, REAL)
        to_testd = np.zeros(ncomp, REAL)


        lee.multipole_exp(sph, 1.0, to_test)
        lee.py_multipole_exp(sph, 1.0, correct)
        lee2.dot_vec_multipole(sph, 1.0, to_testd, to_testm)

        err = np.linalg.norm(to_test - correct, np.inf)
        assert err < 10.**-14
        err = np.linalg.norm(to_testm - correct, np.inf)
        assert err < 10.**-14

        lee2.py_dot_vec(sph, 1.0, correctd)
        
        err = np.linalg.norm(to_testd - correctd, np.inf)
        assert err < 10.**-14




def test_c_ephemeral_harm_1():

    L = 12
    N = 100

    correct_gen = SphGen(L-1, '_A', 'thetaA', 'phiA')
    to_test_gen = SphGenEphemeral(L-1, '_B', 'thetaB', 'phiB')
    
    d = {}
    for lx in range(L):
        for mx in range(-lx, lx+1):
            d[(lx, mx)] = ('err = MAX(ABS({} - {}), err);'.format(
                correct_gen.get_y_sym(lx, mx)[0],
                to_test_gen.get_y_sym(lx, mx)[0],
            ) + 'err = MAX(ABS({} - {}), err);'.format(
                correct_gen.get_y_sym(lx, mx)[1],
                to_test_gen.get_y_sym(lx, mx)[1],
            ),)

    m = to_test_gen(d)

    src = """
    #include <math.h>
    #define ABS(x) ((x) > (0) ? (x) : (-(x)))
    #define MAX(x, y) ((x) > (y) ? (x) : (y))

    extern "C"
    int test(
        const double thetaA,
        const double phiA,
        const double thetaB,
        const double phiB,
        double * err_out
    ){{
        double err = 0.0;
        {CORRECT_GEN}
        // ---------------
        {TO_TEST_GEN}

        *err_out = err;

        return 0;
    }}


    """.format(
        CORRECT_GEN=correct_gen.module,
        TO_TEST_GEN=m
    )
    
    lib = simple_lib_creator(header_code='', src_code=src)['test']

    rng = np.random.RandomState(149135315)

    for testx in range(N):
    
        p = rng.uniform(low=0, high=6, size=2)
        err = ctypes.c_double(0)

        lib(
            ctypes.c_double(p[0]),
            ctypes.c_double(p[1]),
            ctypes.c_double(p[0]),
            ctypes.c_double(p[1]),
            ctypes.byref(err)
        )

        assert abs(err.value) < 10.**-15




def cube_ind(L, M):
    return ((L) * ( (L) + 1 ) + (M) )


def test_c_ephemeral_harm_lexp_1():

    L = 12
    N = 100
    ncomp = L*L*2

    to_test_gen = SphGenEphemeral(L-1, '_B', 'theta', 'phi', radius_symbol='rhol')
    
    radius_gen = 'const double iradius = 1.0 / radius;\n'
    radius_gen += 'const double {} = iradius;\n'.format(to_test_gen.get_radius_sym(0))
    for lx in range(1, L):
        radius_gen += 'const double {} = {} * iradius;'.format(
            to_test_gen.get_radius_sym(lx),
            to_test_gen.get_radius_sym(lx-1)
        )


    d = {}

    for lx in range(L):
        for mx in range(-lx, lx+1):
            d[(lx, mx)] = (
                'inner_out[{ind}] += {ylmm};'.format(
                    ind=cube_ind(lx, mx),
                    ylmm=str(to_test_gen.get_y_sym(lx, -mx)[0]),
                    l=lx
                ),
                'inner_out[{ind}] += {ylmm};'.format(
                    ind=cube_ind(lx, mx) + L*L,
                    ylmm=str(to_test_gen.get_y_sym(lx, -mx)[1]),
                    l=lx
                )
            )


    m = to_test_gen(d)

    src = """
    #include <math.h>
    #define NCOMP {NCOMP}
    #define ABS(x) ((x) > (0) ? (x) : (-(x)))
    #define MAX(x, y) ((x) > (y) ? (x) : (y))

    extern "C"
    int test(
        const int N,
        const double * RESTRICT vec_radius,
        const double * RESTRICT vec_theta,
        const double * RESTRICT vec_phi,
        double * RESTRICT out
    ){{
        
        double inner_out[NCOMP];
        for(int ix=0 ; ix<NCOMP ; ix++){{
            inner_out[ix] = 0.0;
        }}

        for(int ix=0 ; ix<N ; ix++){{
            const double radius = vec_radius[ix];
            const double theta = vec_theta[ix];
            const double phi = vec_phi[ix];

            {RADIUS_GEN}
            {TO_TEST_GEN}

        }}
        for(int ix=0 ; ix<NCOMP ; ix++){{
            out[ix] = inner_out[ix];
        }}

        return 0;
    }}


    """.format(
        NCOMP=ncomp,
        RADIUS_GEN=radius_gen,
        TO_TEST_GEN=m
    )
    

    lib = simple_lib_creator(header_code='', src_code=src)['test']

    rng = np.random.RandomState(149135315)


    set_radius = np.array(rng.uniform(1.0, 4.0, N), ctypes.c_double)
    set_theta = np.array(rng.uniform(0.0, 6.3, N), ctypes.c_double)
    set_phi = np.array(rng.uniform(0.0, 3.1, N), ctypes.c_double)
    
    to_test = np.zeros(ncomp, ctypes.c_double)

    lib(
        ctypes.c_int(N),
        set_radius.ctypes.get_as_parameter(),
        set_theta.ctypes.get_as_parameter(),
        set_phi.ctypes.get_as_parameter(),
        to_test.ctypes.get_as_parameter()
    )


    correct = np.zeros(ncomp, ctypes.c_double)

    for tx in range(N):
        py_local_exp(
            L,
            (
                set_radius[tx],
                set_theta[tx],
                set_phi[tx]
            ),
            1.0,
            correct
        )


    err = np.linalg.norm(correct - to_test, np.inf)

    #print(err)

    #print(correct)
    #print("----------------")
    #print(to_test)

    assert err < 10.**-13













