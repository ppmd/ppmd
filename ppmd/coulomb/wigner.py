from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import numpy as np
import scipy
from scipy.special import jacobi, binom
import math
import cmath



from functools import wraps


class _old_cached(object):
    def __init__(self, maxsize=None):
        pass
    def __call__(self, function):
        # pythontips.com
        memo = {}
        @wraps(function)
        def wrapper(*args):
            if args in memo:
                return memo[args]
            else:
                rv = function(*args)
                memo[args] = rv
                return rv
        return wrapper

try:
    from functools import lru_cache
    cached = lru_cache
except Exception as e:

    cached = _old_cached

@cached(maxsize=32)
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


@cached(maxsize=40960)
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

@cached(maxsize=1024)
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

@cached(maxsize=1024)
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
            coeff *= eps_m(1*m)
            coeff *= eps_m(mp)
            out.real[mpx, mx] = coeff
    return out

@cached(maxsize=1024)
def R_zy(p, alpha, beta):
    return np.matmul(R_y(p,beta), R_z(p,alpha))

@cached(maxsize=1024)
def R_zyz(p, alpha, beta, gamma):
    return np.matmul(R_z(p,gamma),
           np.matmul(R_y(p,beta), R_z(p,alpha)))





