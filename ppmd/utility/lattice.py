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

    Assumes n0, n1, n2 are greater than 1.

    :param n: (n0, n1, n2) Number of unit images in each dimension.
    :param e: (e0, e1, e2) Extent of domain in each dimension.
    :return: Numpy array of points.
    """

    assert len(n) == 3, "n should be length 3"
    assert len(e) == 3, "e should be length 3"

    n = (int(n[0]), int(n[1]), int(n[2]))
    e = (float(e[0]), float(e[1]), float(e[2]))

    w = np.array((e[0]/float(n[0]), e[1]/float(n[1]), e[2]/float(n[2])),
                 dtype=ctypes.c_double)


    starts = np.array((-0.5*e[0] + 0.5*w[0],
                      -0.5*e[1] + 0.5*w[1],
                      -0.5*e[2] + 0.5*w[2]),
                      dtype=ctypes.c_double)


    unit = np.array(
        ((0.0, 0.0, 0.0),
        (0.5*w[0], 0.5*w[1], 0.0*w[2]),
        (0.5*w[0], 0.0*w[1], 0.5*w[2]),
        (0.0*w[0], 0.5*w[1], 0.5*w[2])),
        dtype=ctypes.c_double
    )

    nt = n[0]*n[1]*n[2]

    arr = np.zeros([nt*4, 3], dtype=ctypes.c_double)

    tmp = np.zeros_like(unit)

    for iz in xrange(n[2]):
        for iy in xrange(n[1]):
            for ix in xrange(n[0]):
                i = iz*(n[0]*n[1]) + iy*n[0] + ix
                tmp.fill(0.0)

                tmp[:,0] = w[0]*ix + unit[:,0] + starts[0]
                tmp[:,1] = w[1]*iy + unit[:,1] + starts[1]
                tmp[:,2] = w[2]*iz + unit[:,2] + starts[2]

                arr[i*4:(i+1)*4:, :] = tmp


    for ix in range(nt*4):
        if arr[ix, 0] >= 0.5*e[0]:
            arr[ix, 0 ] = -0.5*e[0]
        if arr[ix, 1] >= 0.5*e[1]:
            arr[ix, 1 ] = -0.5*e[1]
        if arr[ix, 2] >= 0.5*e[2]:
            arr[ix, 2 ] = -0.5*e[2]

    return arr



















