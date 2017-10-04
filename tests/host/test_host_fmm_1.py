from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)

import ppmd as md
from ppmd.coulomb.fmm import *


MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier()
DEBUG = True


class FakeDomain(object):
    def __init__(self, extent, cart_comm):
        self.comm = cart_comm
        self.extent = extent

def test_fmm_init_1():
    dims = md.mpi.MPI.Compute_dims(MPISIZE, 3)
    if MPIRANK == 0 and DEBUG:
        print("DIMS", dims[::-1])
    cc = md.mpi.create_cartcomm(
        md.mpi.MPI.COMM_WORLD, dims[::-1], (1,1,1), True)

    domain = FakeDomain(extent=10, cart_comm=cc)

    fmm = PyFMM(domain=domain, N=1000)
























