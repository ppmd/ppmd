#!/usr/bin/python
from __future__ import print_function

import pytest
import ctypes
import numpy as np

from ppmd import *
from ppmd.coulomb.octal import *
from ppmd.coulomb.fmm import *

from ppmd.cuda import CUDA_IMPORT

if CUDA_IMPORT:
    from ppmd.cuda import *

cuda = pytest.mark.skipif("CUDA_IMPORT is False")

MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
MPISIZE = mpi.MPI.COMM_WORLD.Get_size()

@cuda
def test_cuda_fmm_1():
    R = 4

    crN = 10
    N = crN**3

    E = 3.*crN

    rc = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2

    ASYNC = False
    free_space = True
    CUDA = True

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda=CUDA)

    rng = np.random.RandomState(seed=1234)

    lx = 3
    fmm.tree_halo[lx][:] = rng.uniform(low=-2.0, high=2.0,
                                       size=fmm.tree_halo[lx].shape)

    fmm._halo_exchange(lx)
    fmm._translate_m_to_l(lx)

    radius = fmm.domain.extent[0] / \
             fmm.tree[lx].ncubes_side_global

    lx_cuda = fmm._cuda_mtl.translate_mtl(fmm.tree_halo, lx, radius)

    for px in range(lx_cuda.ravel().shape[0]):
        assert abs(fmm.tree_plain[lx].ravel()[px] - lx_cuda.ravel()[px]) < \
                10.** -12



@cuda
def test_cuda_fmm_2():
    R = 4

    crN = 10
    N = crN**3

    E = 3.*crN

    rc = 10.

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2

    ASYNC = False
    free_space = '27'
    CUDA = True

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda=CUDA)

    rng = np.random.RandomState(seed=1234)

    lx = 3
    fmm.tree_halo[lx][:] = rng.uniform(low=-2.0, high=2.0,
                                       size=fmm.tree_halo[lx].shape)

    fmm._halo_exchange(lx)
    fmm._translate_m_to_l(lx)

    radius = fmm.domain.extent[0] / \
             fmm.tree[lx].ncubes_side_global

    lx_cuda = fmm._cuda_mtl.translate_mtlz(fmm.tree_halo, lx, radius)

    for jx in range(fmm.L):
        print(jx)
        for kx in range(-jx, jx+1):
            print("{: 2d} | {: .8f} {: .8f} | {: .8f} {: .8f} ".format(
                kx,
                fmm.tree_plain[lx][1,1,1,fmm.re_lm(jx,kx)],
                lx_cuda[1,1,1,fmm.re_lm(jx,kx)],
                fmm.tree_plain[lx][1,1,1,fmm.im_lm(jx,kx)],
                lx_cuda[1,1,1,fmm.im_lm(jx,kx)]))

    assert np.linalg.norm(lx_cuda.ravel() - fmm.tree_plain[lx].ravel(),
                          np.inf) < 10.**-14

    fmm.tree_plain[lx][:] = 0.0
    fmm._translate_m_to_l_cart(lx)

    assert np.linalg.norm(lx_cuda.ravel() - fmm.tree_plain[lx].ravel(),
                          np.inf) < 10.**-14
