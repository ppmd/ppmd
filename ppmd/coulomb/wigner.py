from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import numpy as np
import scipy
from scipy.special import jacobi, binom
import math
import cmath
import ctypes

from ppmd.coulomb.cached import cached
from ppmd.lib.build import simple_lib_creator

import os
_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

@cached(maxsize=1024)
def wigner_d(j, mp, m, beta):
    """
    Compute the Wigner d-matrix d_{m', m}^j(\beta) using Jacobi polynomials.
    Taken from wikipedia which claims Wigner, E.P. 1931 as a source. Matches
    the recursion based method in wigner_d_rec.
    """

    j = int(j)
    mp = int(mp)
    m = int(m)

    k = min(j+m, j-m, j+mp, j-mp)

    if k == j + m:
        a = mp - m; l = mp - m
    elif k == j - m:
        a = m - mp; l = 0
    elif k == j + mp:
        a = m - mp; l = 0
    elif k == j - mp:
        a = mp - m; l = mp - m
    else:
        raise RuntimeError("could not compute a,l")

    b = 2 * (j - k) - a

    return ((-1.)**l) * (binom(2*j - k, k + a)**0.5) * \
           (binom(k+b, b)**-0.5) * (math.sin(0.5*beta)**a) * \
           (math.cos(0.5*beta)**b) * jacobi(k,a,b)(math.cos(beta))


@cached(maxsize=4096000)
def wigner_d_rec(j, mp, m, beta):
    """
    Compute the Wigner d-matrix d_{m', m}^j(\beta) using recursion relations
    Uses recursion relations in:

    "A fast and stable method for rotating spherical harmonic expansions",
    Z. Gimbutas, L.Greengard

    Corrections:
    Equation (11), last term, numerator in square root should be:
    (n-m)(n-m-1) not (n-m)(n-m+1) to match the Jacobi Polynomial version.
    """

    j = int(j)
    mp = int(mp)
    m = int(m)

    # base cases
    if j == 0 and mp == 0 and m == 0:
        return 1.0
    elif j < 0:
        raise RuntimeError("negative j is invalid")
    elif abs(m) > j or abs(mp) > j:
        return 0.0


    # 3rd Equation 11
    denom = ((j+mp)*(j-mp))
    if denom != 0:
        cb = math.cos(0.5*beta)
        sb = math.sin(0.5*beta)
        sc = sb * cb

        term1 = sc * math.sqrt((j+m)*(j+m-1)/denom) * \
            wigner_d_rec(j-1, mp, m-1, beta)

        term2 = (cb*cb - sb*sb)*math.sqrt(
            ((j-m)*(j+m))/denom) * \
            wigner_d_rec(j-1, mp, m, beta)

        term3 = sc * math.sqrt(
            (j-m)*(j-m-1)/denom) * \
            wigner_d_rec(j-1, mp, m+1, beta)

        return term1 + term2 - term3


    # 1st Equation 9
    denom = ((j + mp)*(j + mp -1))
    if denom != 0:
        term1 = (math.cos(0.5*beta)**2.) * math.sqrt(
            ((j + m)*(j + m - 1))/denom) * \
            wigner_d_rec(j-1,mp-1,m-1,beta)

        term2 = 2. * math.sin(0.5*beta)*math.cos(0.5*beta) * math.sqrt(
            ((j+m)*(j-m))/denom) * \
            wigner_d_rec(j-1,mp-1,m, beta)

        term3 = (math.sin(0.5*beta)**2.) * math.sqrt(
            ((j-m)*(j-m-1.))/denom) * \
            wigner_d_rec(j-1, mp-1, m+1, beta)

        return term1 - term2 + term3

    # 2nd Equation 10
    denom = ((j-mp)*(j-mp-1))
    if denom != 0:
        term1 = (math.sin(0.5*beta)**2.) * math.sqrt(
            ((j+m)*(j+m-1))/denom) * \
            wigner_d_rec(j-1,mp+1,m-1,beta)

        term2 = 2. * math.sin(0.5*beta)*math.cos(0.5*beta) * math.sqrt(
            ((j+m)*(j-m))/denom) * \
            wigner_d_rec(j-1,mp+1,m, beta)

        term3 = (math.cos(0.5*beta)**2.) * math.sqrt(
            ((j-m)*(j-m-1))/denom) * \
            wigner_d_rec(j-1, mp+1, m+1, beta)

        return term1 + term2 + term3

    raise RuntimeError("No suitable recursion relation or base case found.")

def eps_m(m):
    if m < 0: return 1.0
    return (-1.)**m

@cached(maxsize=4096)
def R_z(p, x):
    """
    matrix to apply to complex vector of p moments to rotate the basis functions
    used to compute the moments around the z-axis by angle x.
    """
    ncomp = 2*p+1
    out = np.zeros((ncomp, ncomp), dtype=np.complex)
    for mx in range(ncomp):
        m = mx - p
        out[mx, mx] = cmath.exp((1.j) * m * x)
    return out

@cached(maxsize=4096)
def R_z_vec(p, x):
    """
    matrix to apply to complex vector of p moments to rotate the basis functions
    used to compute the moments around the z-axis by angle x.
    """
    ncomp = 2*p+1
    out = np.zeros(ncomp, dtype=np.complex)
    for mx in range(ncomp):
        m = mx - p
        out[mx] = cmath.exp((1.j) * m * x)
    return out

@cached(maxsize=4096)
def R_y(p, x):
    """
    matrix to apply to complex vector of p moments to rotate the basis functions
    used to compute the moments around the y-axis by angle x.
    """
    ncomp = 2*p+1
    out = np.zeros((ncomp, ncomp), dtype=np.complex)
    for mpx in range(ncomp):
        for mx in range(ncomp):
            mp = mpx - p
            m = mx - p
            coeff = wigner_d_rec(p, mp, m, x)
            coeff *= eps_m(m)
            coeff *= eps_m(mp)
            out.real[mpx, mx] = coeff
    return out

@cached(maxsize=4096)
def R_zy(p, alpha, beta):
    return np.matmul(R_y(p,beta), R_z(p,alpha))

@cached(maxsize=4096)
def R_zyz(p, alpha, beta, gamma):
    #return np.matmul(R_z(p, gamma),
    #                 np.matmul(R_y(p, beta), R_z(p, alpha)))

    m0 = R_z_vec(p, gamma)
    m1 = R_y(p, beta)
    m2 = R_z_vec(p, alpha)

    ncomp = 2*p+1
    out = np.zeros((ncomp, ncomp), dtype=np.complex)

    for mx in range(m1.shape[1]):
        out[:, mx] = m1[:, mx] * m2[mx]

    for mx in range(m1.shape[0]):
        out[mx, :] *= m0[mx]

    return out


def R_zyz_given_y(p, alpha, beta, gamma, m1):
    #return np.matmul(R_z(p, gamma),
    #                 np.matmul(R_y(p, beta), R_z(p, alpha)))

    m0 = R_z_vec(p, gamma)
    m2 = R_z_vec(p, alpha)

    ncomp = 2*p+1
    out = np.zeros((ncomp, ncomp), dtype=np.complex)

    for mx in range(m1.shape[1]):
        out[:, mx] = m1[:, mx] * m2[mx]

    for mx in range(m1.shape[0]):
        out[mx, :] *= m0[mx]


    return out

def Rzyz_set(p, alpha, beta, gamma, dtype):
    return Rzyz_set_2(p, alpha, beta, gamma, dtype)

def Rzyz_set_2(p, alpha, beta, gamma, dtype):
    """
    Returns the set of matrices needed to rotate all p moments by beta around
    the y axis.
    """
    pointers_real = np.zeros(p, dtype=ctypes.c_void_p)
    pointers_imag = np.zeros(p, dtype=ctypes.c_void_p)

    wp, wm = _wigner_engine(p, beta, eps_scaled=True)

    matrices = {'real': [], 'imag': []}
    for px in range(p):
        r = R_zyz_given_y(px, alpha, beta, gamma, wm[px])
        matrices['real'].append(np.array(r.real, dtype=dtype))
        pointers_real[px] = matrices['real'][-1].ctypes.data
        matrices['imag'].append(np.array(r.imag, dtype=dtype))
        pointers_imag[px] = matrices['imag'][-1].ctypes.data

    return pointers_real, pointers_imag, matrices

def Ry_set(p, beta, dtype):
    """
    Returns the set of matrices needed to rotate all p moments by beta around
    the y axis.
    """
    assert dtype is ctypes.c_double
    wp, wm = _wigner_engine(p, beta, eps_scaled=True)

    return wp, wm

def Rzyz_set_orig(p, alpha, beta, gamma, dtype):
    """
    Returns the set of matrices needed to rotate all p moments by beta around
    the y axis.
    """
    pointers_real = np.zeros(p, dtype=ctypes.c_void_p)
    pointers_imag = np.zeros(p, dtype=ctypes.c_void_p)

    matrices = {'real': [], 'imag': []}
    for px in range(p):
        r = R_zyz(px, alpha, beta, gamma)
        matrices['real'].append(np.array(r.real, dtype=dtype))
        pointers_real[px] = matrices['real'][-1].ctypes.data
        matrices['imag'].append(np.array(r.imag, dtype=dtype))
        pointers_imag[px] = matrices['imag'][-1].ctypes.data

    return pointers_real, pointers_imag, matrices



class _WignerEngine(object):
    def __init__(self):
        with open(str(_SRC_DIR) + \
                          '/FMMSource/WignerSource.cpp') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/WignerSource.h') as fh:
            hpp = fh.read()

        self._lib = simple_lib_creator(hpp, cpp,
            'wigner_matrix')['get_matrix_set']

    def __call__(self, maxj, beta, eps_scaled=False):

        pointers = np.zeros(maxj, dtype=ctypes.c_void_p)
        matrices = []
        
        # places the matrices next to each other in memory
        # may help with smaller ones
        s = 4
        for jx in range(maxj):
            p = 2*jx + 1
            s += p*p
        mat = np.zeros(s, dtype=ctypes.c_double)

        s = 0
        for jx in range(maxj):
            p = 2*jx + 1
            matrices.append(np.reshape(mat[s:s+p*p:].view(), (p,p)))
            pointers[jx] = mat[s::].ctypes.data
            s += p*p

        matrices.append(mat)


        self._lib(
            ctypes.c_int32(maxj),
            ctypes.c_double(beta),
            pointers.ctypes.get_as_parameter()
        )
        if eps_scaled:
            for jx in range(maxj):
                ncomp = 2*jx + 1
                for mpx in range(ncomp):
                    for mx in range(ncomp):
                        mp = mpx - jx
                        m = mx - jx
                        coeff = eps_m(m)
                        coeff *= eps_m(mp)
                        matrices[jx][mpx, mx] *= coeff

        return pointers, matrices

_wigner_engine=_WignerEngine()














































