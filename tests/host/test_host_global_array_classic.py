#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md

N = 16
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()

GlobalArray = md.data.GlobalArray
ParticleDat = md.data.ParticleDat
Kernel = md.kernel.Kernel
ParticleLoop = md.loop.ParticleLoop
ParticleLoopOMP = md.loop.ParticleLoopOMP
Header = md.kernel.Header

from ppmd.access import *

N1 = 4


@pytest.fixture()
def DGAN1():
    return GlobalArray(size=N1, dtype=ctypes.c_double)
@pytest.fixture()
def IGAN1():
    return GlobalArray(size=N1, dtype=ctypes.c_int)

def test_host_global_array_type():
    A = GlobalArray()
    assert type(A) is md.data.GlobalArrayClassic, "bad instance type"



@pytest.mark.skip
def test_host_global_array_1(IGAN1):

    A = IGAN1

    assert type(IGAN1) is md.data.GlobalArrayClassic, "bad instance type"

    A.set(1)
    for ix in range(N1):
        assert A[ix] == 1, "GlobalArray.set 1 failed"
    rint = 4
    A.set(rint)
    for ix in range(N1):
        assert A[ix] == rint, "GlobalArray.set 2 failed"

@pytest.mark.skip
def test_host_global_array_2(DGAN1):
    A = DGAN1
    A.set(1.5)
    for ix in range(N1):
        assert A[ix] == 1.5, "GlobalArray.set 1 failed"
    rf = 3.1415
    A.set(rf)
    for ix in range(N1):
        assert A[ix] == rf, "GlobalArray.set 2 failed"

@pytest.mark.skip
def test_host_global_array_2_5(IGAN1):
    A = IGAN1
    A.set(0)

    A[:] += 1

    for ix in range(N1):
        assert A[ix] == nproc, "GlobalArray.reduction 1 failed"

    A[:] += 1

    for ix in range(N1):
        assert A[ix] == nproc*(nproc+1), "GlobalArray.reduction 2 failed"

    A.set(nproc)
    for ix in range(N1):
        assert A[ix] == nproc, "GlobalArray.set failed"

    A.set(0)
    A[0:3:] += 1
    for ix in range(3):
        assert A[ix] == nproc, "GlobalArray.reduction 1 failed"
    assert A[3] == 0, "reduction failed"

    A.set(0)
    A[:] += 1
    for ix in range(4):
        assert A[ix] == nproc, "GlobalArray.reduction 1 failed"

    A[0:3:] += 1
    for ix in range(3):
        assert A[ix] == nproc*(nproc+1), "GlobalArray.reduction 1 failed"
    assert A[3] == nproc, "reduction failed"


@pytest.mark.skip
def test_host_global_array_3(IGAN1):
    A = IGAN1
    A.set(0)
    A[:] += rank

    csum = np.sum(np.arange(nproc))

    for ix in range(N1):
        assert A[ix] == csum, "GlobalArray.reduction failed"

@pytest.mark.skip
def test_host_global_array_3_5():
    A = GlobalArray(size=nproc, dtype=ctypes.c_int)
    A.set(0)
    A[rank] += 1

    for ix in range(nproc):
        assert A[ix] == 1, "GlobalArray.reduction failed"

    A[rank] += 0

    for ix in range(nproc):
        assert A[ix] == 1, "GlobalArray.reduction failed"

    A[nproc - 1 - rank] += 1

    for ix in range(nproc):
        assert A[ix] == 2, "GlobalArray.reduction failed"

@pytest.mark.skip
def test_host_global_array_4(DGAN1):
    A = DGAN1
    A.set(0)
    A[:] += rank*0.234
    
    fac = sum(range(nproc))
    csuma = 0.234 * fac

    for ix in range(N1):
        assert abs(A[ix] - csuma)<10.**-15, "GlobalArray.reduction 1 failed"

    A[:] += rank*0.234
    csumb = nproc * csuma + fac*0.234

    for ix in range(N1):
        assert abs(A[ix] - csumb)<10.**-15, "GlobalArray.reduction 2 failed"

    A.set(0)

    coeff = 1./(1+np.arange(N1))

    tmp = np.array([ix*rank for ix in coeff])
    A[:] = tmp[:]
    csuma = [coeff[rx]*fac for rx in range(N1)]

    for ix in range(N1):
        assert abs(A[ix] - csuma[ix])<10.**-15, "GlobalArray.reduction 3 failed"


def test_host_global_array_5(DGAN1):
    A = DGAN1
    A.set(0)

    PD = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_int)
    PD[:,0] = np.arange(N1)

    kernel_src = '''
    A[PD.i[0]] = 1;
    '''
    kernel = Kernel('DGAN1', kernel_src, headers=Header('stdio.h'))
    loop = ParticleLoop(kernel=kernel, dat_dict={'A': A(INC), 'PD':PD(READ)})
    loop.execute()

    for ix in range(N1):
        assert abs(A[ix] - nproc)<10.**-15, "GlobalArray.reduction 1 failed"

def test_host_global_array_6(DGAN1):
    A = DGAN1
    A.set(1)

    PD = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_int)
    PD[:,0] = np.arange(N1)

    kernel_src = '''
    '''
    kernel = Kernel('DGAN2', kernel_src)
    loop = ParticleLoop(kernel=kernel, dat_dict={'A': A(INC_ZERO), 'PD':PD(READ)})
    loop.execute()

    for ix in range(N1):
        assert abs(A[ix])<10.**-15, "GlobalArray.reduction 1 failed"

def test_host_global_array_7(DGAN1):
    A = DGAN1
    A.set(1)

    PD = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_int)
    PD[:,0] = np.arange(N1)

    kernel_src = '''
    A[PD.i[0]] = 2;
    '''
    kernel = Kernel('DGAN2', kernel_src)
    loop = ParticleLoop(kernel=kernel, dat_dict={'A': A(INC_ZERO), 'PD':PD(READ)})
    loop.execute()

    for ix in range(N1):
        assert abs(A[ix] - 2*nproc)<10.**-15, "GlobalArray.reduction 1 failed"


def test_host_global_array_8(DGAN1):
    A = DGAN1
    A.set(1)

    PD = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_int)
    PD[:,0] = np.arange(N1)
    PD2 = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_double)
    PD2[:,0] = 100.0

    kernel_src = '''
    A[PD.i[0]] = 2;
    '''
    kernel = Kernel('DGAN2', kernel_src)
    loop = ParticleLoop(kernel=kernel, dat_dict={'A': A(INC_ZERO), 'PD':PD(READ)})
    loop.execute()


    for ix in range(N1):
        assert abs(A[ix] - 2*nproc)<10.**-15, "GlobalArray.reduction 1 failed"

    kernel_src = '''
    PD2.i[0] = A[PD.i[0]];
    '''
    kernel = Kernel('DGAN2', kernel_src)
    loop = ParticleLoop(kernel=kernel, dat_dict={'A': A(READ), 'PD':PD(READ), 'PD2': PD2(WRITE)})
    loop.execute()

    for ix in range(N1):
        assert abs(PD2[ix,] - 2*nproc)<10.**-15, "GlobalArray.reduction 2 failed"


    for ix in range(N1):
        assert abs(A[ix] - 2*nproc)<10.**-15, "GlobalArray.reduction 3 failed"



def test_host_global_array_10():
    A = GlobalArray(size=1, dtype=ctypes.c_double)
    A.set(0.0)

    PD = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_int)
    PD[:,0] = np.arange(N1)
    PD2 = ParticleDat(npart=N1, ncomp=1, dtype=ctypes.c_double)
    PD2[:,0] = -10.0

    kernel_src = '''
    A[0] += 1;
    '''
    kernel = Kernel('DGAN2', kernel_src, headers=Header('stdio.h'))
    loop = ParticleLoopOMP(kernel=kernel, dat_dict={'A': A(INC), 'PD':PD(READ)})
    loop.execute()

    assert abs(A[0] - nproc*4)<10.**-15, "GlobalArray.reduction 1 failed"


    A.set(1)
    kernel_src = '''
    A[0] += 0.0;
    '''
    kernel = Kernel('DGAN2', kernel_src, headers=Header('stdio.h'))
    loop = ParticleLoopOMP(kernel=kernel, dat_dict={'A': A(INC_ZERO), 'PD':PD(READ)})
    loop.execute()

    assert abs(A[0])<10.**-15, "GlobalArray.reduction 1 failed"



def test_host_global_array_11():
    A = GlobalArray(size=1, dtype=ctypes.c_double)
    A.set(1.0)

    N2 = 100
    PD = ParticleDat(npart=N2, ncomp=1, dtype=ctypes.c_int)

    kernel_src = '''
    A[0] += 0;
    '''

    kernel = Kernel('DGAN2', kernel_src, headers=Header('stdio.h'))
    loop = ParticleLoop(kernel=kernel, dat_dict={'A': A(INC), 'PD':PD(READ)})

    loop.execute()

    assert abs(A[0] - 1.0)<10.**-15, "GlobalArray.reduction 1 failed"


def test_host_global_array_12():
    A = GlobalArray(size=1, dtype=ctypes.c_double)
    A.set(1.0)

    N2 = 100
    PD = ParticleDat(npart=N2, ncomp=1, dtype=ctypes.c_int)

    kernel_src = '''
    A[0] += 0;
    '''

    kernel = Kernel('DGAN2', kernel_src, headers=Header('stdio.h'))
    loop = ParticleLoopOMP(kernel=kernel, dat_dict={'A': A(INC), 'PD':PD(READ)})

    loop.execute()

    assert abs(A[0] - 1.0)<10.**-15, "GlobalArray.reduction 1 failed"










