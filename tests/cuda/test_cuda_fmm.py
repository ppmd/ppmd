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

# some MPI COMM collective is broken
@pytest.mark.skipif("True")
@pytest.mark.xfail
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

    lx_cuda = fmm._cuda_mtl.translate_mtl_cart(fmm.tree_halo, lx, radius)

    for px in range(lx_cuda.ravel().shape[0]):
        assert abs(fmm.tree_plain[lx].ravel()[px] - lx_cuda.ravel()[px]) < \
                10.** -12
    fmm.free()


# some MPI COMM collective is broken
@pytest.mark.skipif("True")
@pytest.mark.xfail
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


    assert np.linalg.norm(lx_cuda.ravel() - fmm.tree_plain[lx].ravel(),
                          np.inf) < 10.**-14

    fmm.tree_plain[lx][:] = 0.0
    fmm._translate_m_to_l_cart(lx)

    assert np.linalg.norm(lx_cuda.ravel() - fmm.tree_plain[lx].ravel(),
                          np.inf) < 10.**-14

    fmm.free()


# some MPI COMM collective is broken
@pytest.mark.skipif("True")
def test_cuda_local_1():
    R = 4

    crN = 10
    N = crN**3

    E = 3.*crN

    rc = 10.

    A = state.State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2
    
    A.p = data.PositionDat(ncomp=3)
    A.f = data.ParticleDat(ncomp=3)
    A.q = data.ParticleDat(ncomp=1)
    A.u = data.ParticleDat(ncomp=1)
    A.fc = data.ParticleDat(ncomp=3)
    A.uc = data.ParticleDat(ncomp=1)   
    if MPIRANK == 0:
        A.p[:N:,:] = np.random.uniform(low=-0.499*E, high=0.499*E, size=(N,3))
        A.q[:N:] = np.random.uniform(low=-2.0, high=2.0, size=(N,1))
        bias = np.sum(A.q[:N:])
        A.q[:N:] -= bias/N
    
    A.scatter_data_from(0)


    free_space = False

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda_local=False)
    fmmc = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda_local=True)

    
    p = fmm(positions=A.p, charges=A.q, forces=A.f, potential=A.u)
    pc = fmmc(positions=A.p, charges=A.q, forces=A.fc, potential=A.uc)
    
    assert abs(p - pc) < 10.**-12
    
    nloc = A.npart_local
    err1 = np.linalg.norm(A.f[:nloc:, :] - A.fc[:nloc:,:], np.inf)
    err2 = np.linalg.norm(A.u[:nloc:, :] - A.uc[:nloc:,:], np.inf)
    
    assert err1 < 10**-12
    assert err2 < 10**-12

    fmm.free()
    fmmc.free()


# some MPI COMM collective is broken
@pytest.mark.skipif("True")
def test_cuda_local_2():
    R = 4

    crN = 20
    N = crN**3

    E = 3.*crN

    rc = 10.

    A = state.State()
    A.npart = N
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    eps = 10.**-2
    
    A.p = data.PositionDat(ncomp=3)
    A.f = data.ParticleDat(ncomp=3)
    A.q = data.ParticleDat(ncomp=1)
    A.u = data.ParticleDat(ncomp=1)
    A.fc = data.ParticleDat(ncomp=3)
    A.uc = data.ParticleDat(ncomp=1)   
    if MPIRANK == 0:
        A.p[:N:,:] = np.random.uniform(low=-0.499*E, high=0.499*E, size=(N,3))
        A.q[:N:] = np.random.uniform(low=-2.0, high=2.0, size=(N,1))
        bias = np.sum(A.q[:N:])
        A.q[:N:] -= bias/N
    
    A.scatter_data_from(0)


    free_space = False

    fmm = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda_local=False)
    fmmc = PyFMM(domain=A.domain, r=R, eps=eps, free_space=free_space,
                cuda_local=True)

    
    p = fmm(positions=A.p, charges=A.q, forces=A.f, potential=A.u)
    pc = fmmc(positions=A.p, charges=A.q, forces=A.fc, potential=A.uc)
    
    assert abs(p - pc) < 10.**-12
    
    nloc = A.npart_local
    err1 = np.linalg.norm(A.f[:nloc:, :] - A.fc[:nloc:,:], np.inf)
    err2 = np.linalg.norm(A.u[:nloc:, :] - A.uc[:nloc:,:], np.inf)
    
    assert err1 < 10**-11
    assert err2 < 10**-11

    fmm.free()
    fmmc.free()    







