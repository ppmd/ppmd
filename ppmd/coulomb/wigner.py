from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import numpy as np
import scipy
from scipy.special import jacobi, binom
import math
import cmath



class _idcache(object):
    def __init__(self, maxsize=None):
        return
    def __call__(self, func):
        return func

try:
    from functools import lru_cache
    cached = lru_cache
except Exception as e:
    cached = _idcache


@cached(maxsize=32)
def wigner_d(j, mp, m, beta):
    """
    Compute the Wigner d-matrix d_{m', m}^j(\beta)
    :param j:
    :param mp:
    :param m:
    :param beta:
    """
    k = min(j+m, j-m, j+mp, j-mp)
    a = None
    l = None

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












