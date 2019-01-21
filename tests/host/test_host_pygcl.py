from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

from ppmd import pygcl

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
DEBUG = False


def test1():
    dims = MPI.Compute_dims(MPISIZE, 3)
    if MPIRANK == 0 and DEBUG:
        print("DIM", dims)
    cc = MPI.COMM_WORLD.Create_cart(dims, (1,1,1), True)
    top = cc.Get_topo()[2]

    N = 64
    dtype = ctypes.c_double

    grid_size = (N, N, N)
    rng = np.random.RandomState(seed=3857293)
    orig = np.array(rng.uniform(0., 100., size=grid_size), dtype=dtype)

    np0 = int(math.floor(grid_size[0]/dims[0]))
    np1 = int(math.floor(grid_size[1]/dims[1]))
    np2 = int(math.floor(grid_size[2]/dims[2]))

    s0 = top[0]*np0
    s1 = top[1]*np1
    s2 = top[2]*np2

    e0 = (top[0]+1)*np0 if top[0] < dims[0]-1 else N
    e1 = (top[1]+1)*np1 if top[1] < dims[1]-1 else N
    e2 = (top[2]+1)*np2 if top[2] < dims[2]-1 else N

    works_size = (e0 - s0 + 2,
                  e1 - s1 + 2,
                  e2 - s2 + 2)

    he = pygcl.HaloExchange3D(cc, works_size)

    a = np.zeros(shape=list(works_size) + [1], dtype=dtype)
    a[1:-1:, 1:-1:, 1:-1, 0] = orig[s0:e0:, s1:e1:, s2:e2:]

    # shell should be zero and interior should be the copied data
    for iz in range(works_size[0]):
        for iy in range(works_size[1]):
            for ix in range(works_size[2]):
                # if shell
                if (iz == 0 or iz == works_size[2]-1) and \
                        (iy == 0 or iy == works_size[1]-1) and \
                        (ix == 0 or ix == works_size[0]-1):
                        assert a[iz, iy, ix, 0] == 0

    for iz in range(1, works_size[0]-1):
        for iy in range(1, works_size[1]-1):
            for ix in range(1, works_size[2]-1):
                assert a[iz, iy, ix, 0] == orig[iz+s0-1, iy+s1-1, ix+s2-1]

    # exchange
    he.exchange(a)

    # interior should be unchanged, outer shell should be halo
    for iz in range(works_size[0]):
        for iy in range(works_size[1]):
            for ix in range(works_size[2]):
                assert a[iz, iy, ix, 0] == orig[(iz+s0-1) % grid_size[0],
                                                (iy+s1-1) % grid_size[1],
                                                (ix+s2-1) % grid_size[2]], \
                    "{} {} {}".format(iz, iy, ix)


def test2():

    width = 2

    dims = MPI.Compute_dims(MPISIZE, 3)
    if MPIRANK == 0 and DEBUG:
        print("DIM", dims)
    cc = MPI.COMM_WORLD.Create_cart(dims, (1,1,1), True)
    top = cc.Get_topo()[2]

    N = 64
    M = 2
    dtype = ctypes.c_double

    grid_size = (2*N, N, N)
    rng = np.random.RandomState(seed=3857293)
    orig = np.array(rng.uniform(0., 100., size=list(grid_size)+[M]),
                    dtype=dtype)

    np0 = int(math.floor(grid_size[0]/dims[0]))
    np1 = int(math.floor(grid_size[1]/dims[1]))
    np2 = int(math.floor(grid_size[2]/dims[2]))

    s0 = top[0]*np0
    s1 = top[1]*np1
    s2 = top[2]*np2

    e0 = (top[0]+1)*np0 if top[0] < dims[0]-1 else grid_size[0]
    e1 = (top[1]+1)*np1 if top[1] < dims[1]-1 else grid_size[1]
    e2 = (top[2]+1)*np2 if top[2] < dims[2]-1 else grid_size[2]

    works_size = (e0 - s0 + 2*width,
                  e1 - s1 + 2*width,
                  e2 - s2 + 2*width)

    he = pygcl.HaloExchange3D(cc, works_size, (width, width, width))

    a = np.zeros(shape=list(works_size) + [M], dtype=dtype)
    a[width:-1*width:, width:-1*width:, width:-1*width, :] = \
        orig[s0:e0:, s1:e1:, s2:e2:, :]

    # shell should be zero and interior should be the copied data
    for iz in range(width, works_size[0]-width):
        for iy in range(width, works_size[1]-width):
            for ix in range(width, works_size[2]-width):
                assert np.all(a[iz, iy, ix, :] == orig[
                    iz+s0-width, iy+s1-width, ix+s2-width, :])

    if DEBUG:
        for rx in range(MPISIZE):
            if MPIRANK == rx:
                print("--", MPIRANK)
                print(orig)
                print(a[:,:,:,0])
            MPI.COMM_WORLD.Barrier()

    # exchange
    he.exchange(a)

    if DEBUG:
        for rx in range(MPISIZE):
            if MPIRANK == rx:
                print("--", MPIRANK)
                print('boundary', he.boundary_cells)
                print('halo', he.halo_cells)
                print(s0, e0, s1, e1, s2, e2)
                print(works_size)
                print(orig)
                print(a[:, :, :, 0])
            MPI.COMM_WORLD.Barrier()

    # interior should be unchanged, outer shell should be halo
    for iz in range(works_size[0]):
        for iy in range(works_size[1]):
            for ix in range(works_size[2]):
                assert np.all(a[iz, iy, ix, :] == orig[(iz+s0-width) % grid_size[0],
                                                (iy+s1-width) % grid_size[1],
                                                (ix+s2-width) % grid_size[2], :]),\
                    "{} {} {}".format(iz, iy, ix)



