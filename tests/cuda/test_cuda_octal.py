#!/usr/bin/python

import pytest
import ctypes
import numpy as np

from ppmd import *
from ppmd.coulomb.octal import *

from ppmd.cuda import CUDA_IMPORT

if CUDA_IMPORT:
    from ppmd.cuda import *

cuda = pytest.mark.skipif("CUDA_IMPORT is False")

MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
MPISIZE = mpi.MPI.COMM_WORLD.Get_size()

@cuda
def test_cuda_octal_1():

    dims = mpi.MPI.Compute_dims(MPISIZE, 3)

    nlevels = 6
    ncomp = 10

    cc = mpi.create_cartcomm(
        mpi.MPI.COMM_WORLD, dims[::-1], (1,1,1), True)

    tree = OctalTree(num_levels=nlevels, cart_comm=cc)

    cudadataplain = OctalCudaDataTree(tree=tree, ncomp=ncomp, mode='plain',
                                      dtype=ctypes.c_double)

    dataplain = OctalDataTree(tree=tree, ncomp=ncomp, mode='plain',
                              dtype=ctypes.c_double)

    dataplain2 = OctalDataTree(tree=tree, ncomp=ncomp, mode='plain',
                               dtype=ctypes.c_double)

    rng = np.random.RandomState(seed=1234)

    for lx in range(nlevels):
        ls = tree[lx].local_grid_cube_size
        if ls is not None:
            dataplain[lx][:] = rng.uniform(size=(ls[0], ls[1], ls[2], ncomp))

            cudadataplain[lx] = dataplain

    for lx in range(nlevels):
        ls = tree[lx].local_grid_cube_size
        if ls is not None:
            ddata = cudadataplain[lx]
            ddata = ddata.ravel()
            cudadataplain.get(lx, dataplain2)

            for dx in range(ddata.shape[0]):
                assert abs(ddata[dx] - dataplain[lx].ravel()[dx]) < 10.**-16
                assert abs(ddata[dx] - dataplain2[lx].ravel()[dx]) < 10.**-16

