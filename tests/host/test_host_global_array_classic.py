#!/usr/bin/python

import pytest
import ctypes
import numpy as np
import ppmd as md
import math

N = 16
E = 8.
Eo2 = E/2.

rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()

GlobalArray = md.data.GlobalArrayClassic

N1 = 4

def _not_factorial(N):
    if N==0:
        return 0
    else:
        return math.factorial(N)


@pytest.fixture()
def DGAN1():
    return GlobalArray(size=N1, dtype=ctypes.c_double, shared_memory=False)
@pytest.fixture()
def IGAN1():
    return GlobalArray(size=N1, dtype=ctypes.c_int, shared_memory=False)

def test_host_global_array_type():
    A = GlobalArray(shared_memory=False)
    assert type(A) is md.data.GlobalArrayClassic, "bad instance type"




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

def test_host_global_array_2(DGAN1):
    A = DGAN1
    A.set(1.5)
    for ix in range(N1):
        assert A[ix] == 1.5, "GlobalArray.set 1 failed"
    rf = 3.1415
    A.set(rf)
    for ix in range(N1):
        assert A[ix] == rf, "GlobalArray.set 2 failed"

def test_host_global_array_2_5(IGAN1):
    A = IGAN1
    A.set(0)

    A[:] += 1

    print A[:]

    for ix in range(N1):
        assert A[ix] == nproc, "GlobalArray.reduction 1 failed"

    A[:] += 1

    for ix in range(N1):
        assert A[ix] == nproc*(nproc+1), "GlobalArray.reduction 2 failed"

    A.set(nproc)
    for ix in range(N1):
        assert A[ix] == nproc, "GlobalArray.set failed"


def test_host_global_array_3(IGAN1):
    A = IGAN1
    A.set(0)
    A[:] += rank

    csum = np.sum(np.arange(nproc))

    for ix in range(N1):
        assert A[ix] == csum, "GlobalArray.reduction failed"

def test_host_global_array_3_5():
    A = GlobalArray(size=nproc, dtype=ctypes.c_int, shared_memory=False)
    A.set(0)
    A[rank] += 1

    for ix in range(nproc):
        assert A[ix] == 1, "GlobalArray.reduction failed"


def test_host_global_array_4(DGAN1):
    A = DGAN1
    A.set(0)
    A[:] += rank*0.234

    fac = _not_factorial(nproc-1)
    csuma = 0.234 * fac

    for ix in range(N1):
        assert abs(A[ix] - csuma)<10.**-15, "GlobalArray.reduction failed"

    A[:] += rank*0.234
    csumb = nproc * csuma + fac*0.234

    for ix in range(N1):
        assert abs(A[ix] - csumb)<10.**-15, "GlobalArray.reduction failed"

    A.set(0)

    coeff = 1./(1+np.arange(N1))

    tmp = np.array([ix*rank for ix in coeff])
    A[:] = tmp[:]
    csuma = [coeff[rx]*fac for rx in xrange(N1)]

    for ix in range(N1):
        assert abs(A[ix] - csuma[ix])<10.**-15, "GlobalArray.reduction failed"













