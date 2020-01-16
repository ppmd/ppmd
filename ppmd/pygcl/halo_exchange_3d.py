from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import collections
import numpy as np
from ctypes import c_uint64, c_int64


# create a CellSlice object for easier halo definition.

class CellSlice(object):
    def __getitem__(self, item):
        return item

Slice = CellSlice()

def get_cart_rank_offset(comm, offset):
    dims = comm.Get_topo()[0]
    top = comm.Get_topo()[2]
    t = [(top[i] + offset[i]) % dims[i] for i in (0, 1, 2)]
    return comm.Get_cart_rank(t)


def create_halo_pairs_slice_halo(comm, cell_array, slicexyz, direction,
                                 width):
    """
    Automatically create the pairs of cells for halos. Slices through 
    whole domain including halo cells.
    """
    # TODO SWAP SLICE ORDER TO MATCH NUMPY
    xr = range(0, cell_array[2])[slicexyz[0]]
    yr = range(0, cell_array[1])[slicexyz[1]]
    zr = range(0, cell_array[0])[slicexyz[2]]

    wx = width[2]
    wy = width[1]
    wz = width[0]

    if not isinstance(xr, collections.abc.Iterable):
        xr = [xr]
    if not isinstance(yr, collections.abc.Iterable):
        yr = [yr]
    if not isinstance(zr, collections.abc.Iterable):
        zr = [zr]

    l = len(xr) * len(yr) * len(zr)

    b_cells = np.zeros(l, dtype=c_uint64)
    h_cells = np.zeros(l, dtype=c_uint64)

    i = 0

    for iz in zr:
        for iy in yr:
            for ix in xr:
                b_cells[i] = ix + (iy + iz * cell_array[1]) * cell_array[2]

                _ix = (ix + direction[2] * 2*wx) % cell_array[2]
                _iy = (iy + direction[1] * 2*wy) % cell_array[1]
                _iz = (iz + direction[0] * 2*wz) % cell_array[0]

                h_cells[i] = _ix + (_iy + _iz * cell_array[1]) * cell_array[2]

                i += 1

    recv_rank = get_cart_rank_offset(comm,
        (-1 * direction[0], -1 * direction[1], -1 * direction[2]))
    send_rank = get_cart_rank_offset(comm,
         (direction[0], direction[1], direction[2]))

    return b_cells, h_cells, send_rank, recv_rank


class HaloExchange3D(object):
    def __init__(self, comm, dim, width=(1, 1, 1)):
        """
        Initialise a halo exchange object. Currently assumes periodic boundary
        conditions.
        
        :param comm: Cartesian communicator to use with dimension=3.
        :param dim: Tuple, local domain size plus space for halos. For example
        if the local domain size is 4x4x4 with a halo of size 1, then dim is 
        6x6x6.
        :param width: halo width in each dimension e.g. (1,1,1).
        """

        self.comm = comm
        """Cartesian communicator used."""
        self.dim = dim
        """Dimensions of local domain plus halo regions."""
        self.send_ranks = np.zeros(6, dtype=c_uint64)
        """Tuple 6x1, MPI ranks in self.comm this rank will send to."""
        self.recv_ranks = np.zeros(6, dtype=c_uint64)
        """Tuple 6x1, MPI ranks in self.comm this rank will recv from."""

        wx = width[2]
        wy = width[1]
        wz = width[0]

        # TODO SWAP SLICE ORDER TO MATCH NUMPY

        cell_pairs = (
           # First exchange the halos cannot contain anything useful
           create_halo_pairs_slice_halo(comm, dim,
                Slice[wx:2*wx:, wy:-1*wy:, wz:-wz:], (0, 0, -1), width),
           create_halo_pairs_slice_halo(comm, dim,
                Slice[-2*wx:-1*wx:, wy:-1*wy:, wz:-wz:], (0, 0, 1), width),

           # No point exchanging anything extra in z direction
           create_halo_pairs_slice_halo(comm, dim,
                Slice[::, wy:2*wy, wz:-wz:], (0, -1, 0), width),
           create_halo_pairs_slice_halo(comm, dim,
                Slice[::, -2*wy:-1*wy, wz:-wz:], (0, 1, 0), width),

           # Exchange all halo cells from x and y
           create_halo_pairs_slice_halo(comm, dim, Slice[::, ::, wz:2*wz:],
                                        (-1, 0, 0), width),
           create_halo_pairs_slice_halo(comm, dim, Slice[::, ::, -2*wz:-1*wz:],
                                        (1, 0, 0), width)
        )
        bs = np.zeros(1, dtype=c_uint64)
        b = np.zeros(0, dtype=c_uint64)

        hs = np.zeros(1, dtype=c_uint64)
        h = np.zeros(0, dtype=c_uint64)



        tnb = 1
        tnh = 1

        for hx, bhx in enumerate(cell_pairs):
            # Boundary and Halo start index.
            tnb = max(tnb, len(bhx[0]))
            tnh = max(tnh, len(bhx[1]))

            bs = np.append(bs, np.array(len(bhx[0]) + bs[-1], dtype=c_uint64))
            hs = np.append(hs, np.array(len(bhx[1]) + hs[-1], dtype=c_uint64))

            # Actual cell indices
            b = np.append(b, bhx[0])
            h = np.append(h, bhx[1])

            self.send_ranks[hx] = bhx[2]
            self.recv_ranks[hx] = bhx[3]

        self.boundary_starts = bs
        """Tuple 7x1, if this tuple is labeled b then direction i packs 
        elements c_{b_i} to c_{b_{i+1}} where c is self.boundary cells."""
        self.boundary_cells = b
        """See self.boundary starts, cells to pack from."""
        self.halo_starts = hs
        """See self.bounadry_starts, same but for halo cells to unpack into."""
        self.halo_cells = h
        """See self.boundary_starts, cells to unpack into."""
        self.tmp_ncomp = max(tnb, tnh)

    def exchange(self, array):
        """
        Halo exchange the passed array using the setup scheme.
        :param array: Array size (n2, n1, n0, d) where (n2, n1, n0) is the 
        tuple the halo exchange scheme was constructed with. d is the number of
        components at each point (automatically determined). Data type needs to
        be compatible with mpi4py (automatically determined).
        
        :return: None, Passed array is modified.
        """
        ncomp = array.shape[-1]

        # case if 4d numpy array

        if len(array.shape) == 4:
            d123 = array.shape[0]*array.shape[1]*array.shape[2]
            array = array.view().reshape(d123, ncomp)
        elif len(array.shape) == 2:
            pass
        else:
            raise RuntimeError("unknown array shape passed")

        tmp_space = np.zeros(shape=(self.tmp_ncomp, ncomp), dtype=array.dtype)

        for dir in range(6):
            # pack
            n = self.boundary_starts[dir + 1] - self.boundary_starts[dir]
            tmp_space[0:n:, :] = array[self.boundary_cells[
                self.halo_starts[dir]: self.halo_starts[dir+1]:], :]

            self.comm.Sendrecv_replace(
                tmp_space[0:n:, :], self.send_ranks[dir], 0,
                self.recv_ranks[dir])

            # unpack
            array[self.halo_cells[
                  self.halo_starts[dir]: self.halo_starts[dir+1]:], :
            ] = tmp_space[0:n:, :]





