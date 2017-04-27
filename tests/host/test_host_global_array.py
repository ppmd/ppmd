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

GlobalArray = md.data.GlobalArray

N1 = 256

@pytest.fixture()
def DGAN1():
    return GlobalArray(size=N1, dtype=ctypes.c_double)
@pytest.fixture()
def IGAN1():
    return GlobalArray(size=N1, dtype=ctypes.c_int)

def test_host_global_array_1(IGAN1):
    A = IGAN1
    A.set(1)
    for ix in range(N1):
        assert A[ix] == 1, "GlobalArray.set failed"
    rint = int(np.random.uniform(low=2, high=100, size=1)[0])
    A.set(rint)
    for ix in range(N1):
        assert A[ix] == rint, "GlobalArray.set failed"

def test_host_global_array_2(DGAN1):
    A = DGAN1
    A.set(1.5)
    for ix in range(N1):
        assert A[ix] == 1.5, "GlobalArray.set failed"
    rf = np.random.uniform(low=1, high=100, size=1)[0]
    A.set(rf)
    for ix in range(N1):
        assert A[ix] == rf, "GlobalArray.set failed"

def test_host_global_array_2_5(IGAN1):
    A = IGAN1
    A.set(0)
    A[:] += 1

    for ix in range(N1):
        assert A[ix] == nproc, "GlobalArray.reduction failed"

    A[:] += 1

    for ix in range(N1):
        assert A[ix] == nproc*(nproc+1), "GlobalArray.reduction failed"

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
    A = GlobalArray(size=nproc, dtype=ctypes.c_int)
    A.set(0)
    A[rank] += 1

    for ix in range(nproc):
        assert A[ix] == 1, "GlobalArray.reduction failed"


def test_host_global_array_4(DGAN1):
    A = DGAN1
    A.set(0)
    A[:] += rank*0.234

    fac = math.factorial(nproc-1)
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













