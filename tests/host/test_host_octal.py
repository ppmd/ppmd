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

    def Get_cart_rank(self, index):
        lindex = sum(i*j for i,j in zip(index, self.dims))
        print(index, lindex)
        return lindex

@pytest.fixture(
    scope="module",
    params=(
            ((1,1,1), 1, ([0], [0], [0]), ([[0]], [[0]], [[0]])),
            ((2,2,2), 1, ([1], [1], [1]), ([[0, 1]], [[0, 1]], [[0, 1]])),
            ((3,3,3), 2, ([0,2], [0,2], [0,2]), ( [[0, 1],[1, 2]],
                                                  [[0, 1],[1, 2]],
                                                  [[0, 1],[1, 2]] )
            ),
            ((2,2,1), 2, ([0,1], [0,1], [0,0]), ( [[0],[1]],
                                                  [[0],[1]],
                                                  [[0],[0]] )
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




@pytest.fixture(
    scope="module",
    params=(
            {
                'ncube': 1,
                'owners': (3,),
                'starts': {
                    0: [0, 0],
                    1: [0, 0],
                    2: [0, 0],
                    3: [0, 3]
                },
                'contribs': {
                    0: [],
                    1: [],
                    2: [],
                    3: [0, 1, 2]
                },
                'sends': {
                    0: [3],
                    1: [3],
                    2: [3],
                    3: [-1]
                },
            },
            {
                'ncube': 3,
                'owners': (0, 1, 1, 2, 3, 3, 2, 3, 3, 0, 1, 1, 2, 3, 3, 2, 3,
                           3, 0, 1, 1, 2, 3, 3, 2, 3, 3),
                'starts': {
                    0: [0]*28,
                    1: [0]*2 + [1]*9 + [2]*9 + [3]*8,
                    2: [0]*4 + [1]*9 + [2]*9 + [3]*6,
                    3: [0]*5 + [3, 4, 4] + [5]*6 + [8,9,9] +[10]*6 + \
                       [13, 14, 14, 15, 15]
                },
                'contribs': {
                    0: [],
                    1: [0,0,0],
                    2: [0,0,0],
                    3: [0, 1, 2, 1, 2]*3
                },
                'sends': {
                    0: [-1,  1, -1,  2,  3, -1, -1, -1, -1] * 3,
                    1: [-1, -1, -1, -1,  3,  3, -1, -1, -1] * 3,
                    2: [-1, -1, -1, -1,  3, -1, -1,  3, -1] * 3,
                    3: [-1]*27
                },
            },
    )
)
def cube_size(request):
    return request.param

DEBUG = False

@pytest.mark.skipif("MPISIZE != 4")
def test_octal_cube_owner_map_2(cube_size):

    ncube = cube_size['ncube']
    dims = md.mpi.MPI.Compute_dims(MPISIZE, 3)

    if MPIRANK == 0 and DEBUG:
        print("DIMS", dims[::-1])

    cc = md.mpi.create_cartcomm(
        md.mpi.MPI.COMM_WORLD, dims[::-1], (1,1,1), True)

    o = cube_owner_map(cc, ncube)
    owners, contribs = o.compute_grid_ownership(
        cc.Get_topo()[0], ncube)

    if MPIRANK == 0 and DEBUG:
        print("\nowners | contribs", owners, "|", contribs)
    md.mpi.MPI.COMM_WORLD.Barrier()

    cube_to_mpi = o.compute_map_product_owners(cc, owners)

    if MPIRANK == 0 and DEBUG:
        print("cube_to_mpi", cube_to_mpi)
    md.mpi.MPI.COMM_WORLD.Barrier()

    starts, con_ranks, send_ranks = o.compute_map_product_contribs(
        cc, cube_to_mpi, contribs)

    for rx in range(cc.Get_size()):
        if MPIRANK == rx and DEBUG:
            print(MPIRANK, starts, con_ranks, send_ranks)
        md.mpi.MPI.COMM_WORLD.Barrier()

    for i, ix in enumerate(starts):
        assert ix == cube_size['starts'][MPIRANK][i]
    for i, ix in enumerate(con_ranks):
        assert ix == cube_size['contribs'][MPIRANK][i]
    for i, ix in enumerate(send_ranks):
        assert ix == cube_size['sends'][MPIRANK][i], "{}|{}|{}:{}".format(MPIRANK, ix, i, cube_size['sends'][MPIRANK])























