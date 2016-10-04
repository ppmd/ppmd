"""
Tools to create numpy arrays containing lattices
"""

# system level imports
import numpy as np
import ctypes
import itertools


def nested_iterator(low, high):
    b = [xrange(i,j) for i,j in zip(low, high)]
    return itertools.product(*b)


def cubic_lattice(n, e):
    """
    Create a cubic lattice of points centred on the origin.
    Dimension is determined by the length of the passed tuples.

    :param n: (n0, n1, n2 ...) Number of points in each dimension
    :param e: (e0, e1, e2, ...) Extent of domain in each dimension
    :return: Numpy array of points.
    """

    n = list(n)
    e = list(e)

    assert len(n) == len(e), "Error: Dimension mismatch"

    ndim = len(n)
    n3 = list(n) + [ndim]
    arr = np.zeros(n3, dtype=ctypes.c_double)

    steps = [0] * ndim
    starts = [0] * ndim
    for v in range(ndim):
        steps[v] = float(e[v])/float(n[v])
        starts[v] = -0.5*float(e[v]) + 0.5*steps[v]


    t = [0] * ndim
    for p in nested_iterator([0] * ndim, n):
        for v in range(ndim):
            t[v] = starts[v] + float(p[v]) * steps[v]
            #print p, v, t[v]

        arr[p] = t

    return arr.reshape([np.product(n), ndim])


def fcc(n, e):
    """
    Create an FCC lattice, centred on the origin. Dimension is fixed at three.
    This function generates a lattice by using a base image consisting of a
    corner atom and three atoms in the centres of the adjacent faces.

    :param n: (n0, n1, n2) Number of unit images in each dimension.
    :param e: (e0, e1, e2) Extent of domain in each dimension.
    :return: Numpy array of points.
    """

    assert len(n) == 3, "n should be length 3"
    assert len(e) == 3, "e should be length 3"

















