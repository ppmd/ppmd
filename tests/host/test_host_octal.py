from __future__ import print_function

import pytest
import ctypes
import numpy as np
import ppmd as md

from ppmd.coulomb.octal import *

MPISIZE = md.mpi.MPI.COMM_WORLD.Get_size()
MPIRANK = md.mpi.MPI.COMM_WORLD.Get_rank()


@pytest.fixture
def state():
    A = 1
    return A

class _fake_cartcomm(object):
    def __init__(self, dims):
        self.dims = dims

    def Get_topo(self):
        return (self.dims, self.dims, self.dims)

@pytest.fixture(
    scope="module",
    params=(
            ((1,1,1), 1, ([0], [0], [0]), ([[]], [[]], [[]])),
            ((2,2,2), 1, ([1], [1], [1]), ([[0]], [[0]], [[0]])),
            ((3,3,3), 2, ([0,2], [0,2], [0,2]), ( [[1],[1]],
                                                  [[1],[1]],
                                                  [[1],[1]] )
            ),
            ((2,2,1), 2, ([0,1], [0,1], [0,0]), ( [[],[]],
                                                  [[],[]],
                                                  [[],[]] )
            )
    )
)
def fake_cartcomm(request):
    return request.param

def test_octal_cube_owner_map_1(fake_cartcomm):

    cc = _fake_cartcomm(fake_cartcomm[0])
    cube_count = fake_cartcomm[1]
    expected_owners = fake_cartcomm[2]
    expected_contribs = fake_cartcomm[3]

    o = cube_owner_map(cc, cube_count)

    owners, contribs = o.compute_grid_ownership(
        cc.Get_topo()[0], cube_count)

    for ix, ox in enumerate(owners):
        assert ox == expected_owners[ix]
    for ix, ox in enumerate(contribs):
        assert ox == expected_contribs[ix]


def test_octal_cube_owner_map_2():
    ncube = 2
    dims = md.mpi.MPI.Compute_dims(MPISIZE, 3)

    if MPIRANK == 0:
        print("DIMS", dims)

    cc = md.mpi.create_cartcomm(md.mpi.MPI.COMM_WORLD, dims, (1,1,1), True)
    o = cube_owner_map(cc, ncube)
    owners, contribs = o.compute_grid_ownership(
        cc.Get_topo()[0], ncube)

    if MPIRANK == 0:
        print(owners, contribs)

    cube_to_mpi = o.compute_map_product(cc, owners, contribs)

    if MPIRANK == 0:
        print(cube_to_mpi)

    md.mpi.MPI.COMM_WORLD.Barrier()













