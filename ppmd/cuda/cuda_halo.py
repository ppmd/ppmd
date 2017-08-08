from __future__ import print_function, division, absolute_import
"""
Methods to aid CUDA halo exchanges.
"""

# system level imports
import ctypes, collections
import numpy as np

# package level imports
import ppmd.mpi as mpi
from ppmd import abort

# cuda level imports
from ppmd.cuda import cuda_runtime, cuda_mpi, cuda_cell, cuda_base

class CellSlice(object):
    def __getitem__(self, item):
        return item

Slice = CellSlice()


def create_halo_pairs(domain_in, slicexyz, direction):
    """
    Automatically create the pairs of cells for halos.
    """
    cell_array = domain_in.cell_array
    extent = domain_in.extent
    comm = domain_in.comm
    dims = mpi.cartcomm_dims(comm)
    top = mpi.cartcomm_top(comm)
    periods = mpi.cartcomm_periods(comm)

    xr = range(1, cell_array[0] - 1)[slicexyz[0]]
    yr = range(1, cell_array[1] - 1)[slicexyz[1]]
    zr = range(1, cell_array[2] - 1)[slicexyz[2]]

    if not isinstance(xr, collections.Iterable):
        xr = [xr]
    if not isinstance(yr, collections.Iterable):
        yr = [yr]
    if not isinstance(zr, collections.Iterable):
        zr = [zr]

    l = len(xr) * len(yr) * len(zr)

    b_cells = np.zeros(l, dtype=ctypes.c_int)
    h_cells = np.zeros(l, dtype=ctypes.c_int)

    i = 0

    for iz in zr:
        for iy in yr:
            for ix in xr:
                b_cells[i] = ix + (iy + iz * cell_array[1]) * cell_array[0]

                _ix = (ix + direction[0] * 2) % cell_array[0]
                _iy = (iy + direction[1] * 2) % cell_array[1]
                _iz = (iz + direction[2] * 2) % cell_array[2]

                h_cells[i] = _ix + (_iy + _iz * cell_array[1]) * cell_array[0]

                i += 1

    shift = np.zeros(3, dtype=ctypes.c_double)
    for ix in range(3):
        if top[ix] == 0 and periods[ix] == 1 and direction[ix] == -1:
            shift[ix] = extent[ix]
        if top[ix] == dims[ix] - 1 and periods[ix] == 1 and direction[ix] == 1:
            shift[ix] = -1. * extent[ix]


    return b_cells, h_cells, shift




class CartesianHalo(object):

    def __init__(self, # host_halo=halo.HALOS,
                 occ_matrix=cuda_cell.OCCUPANCY_MATRIX):

        # self._host_halo_handle = host_halo

        self._occ_matrix = occ_matrix
        self._version = -1

        self._init = False

        # vars init
        self._boundary_cell_groups = cuda_base.Array(dtype=ctypes.c_int)
        self._boundary_groups_start_end_indices = cuda_base.Array(ncomp=27, dtype=ctypes.c_int)
        self._halo_cell_groups = cuda_base.Array(dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = cuda_base.Array(ncomp=27, dtype=ctypes.c_int)
        self._boundary_groups_contents_array = cuda_base.Array(dtype=ctypes.c_int)
        self._exchange_sizes = cuda_base.Array(ncomp=26, dtype=ctypes.c_int)

        self._halo_shifts = None
        self._reverse_lookup = None

        # ensure first update
        self._boundary_cell_groups.inc_version(-1)
        self._boundary_groups_start_end_indices.inc_version(-1)
        self._halo_cell_groups.inc_version(-1)
        self._halo_groups_start_end_indices.inc_version(-1)
        self._boundary_groups_contents_array.inc_version(-1)
        self._exchange_sizes.inc_version(-1)


    @property
    def occ_matrix(self):
        """
        Returns the occupancy matrix involved.
        :return:
        """
        return self._occ_matrix


    def _get_pairs(self):
        _cell_pairs = (
            create_halo_pairs(self.occ_matrix.domain, Slice[0,0,0],(-1,-1,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,0,0],(0,-1,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,0,0],(1,-1,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[0,::,0],(-1,0,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,::,0],(0,0,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,::,0],(1,0,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[0,-1,0],(-1,1,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,-1,0],(0,1,-1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,-1,0],(1,1,-1)),

            create_halo_pairs(self.occ_matrix.domain, Slice[0,0,::],(-1,-1,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,0,::],(0,-1,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,0,::],(1,-1,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[0,::,::],(-1,0,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,::,::],(1,0,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[0,-1,::],(-1,1,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,-1,::],(0,1,0)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,-1,::],(1,1,0)),

            create_halo_pairs(self.occ_matrix.domain, Slice[0,0,-1],(-1,-1,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,0,-1],(0,-1,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,0,-1],(1,-1,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[0,::,-1],(-1,0,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,::,-1],(0,0,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,::,-1],(1,0,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[0,-1,-1],(-1,1,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[::,-1,-1],(0,1,1)),
            create_halo_pairs(self.occ_matrix.domain, Slice[-1,-1,-1],(1,1,1))
        )

        _bs = np.zeros(1, dtype=ctypes.c_int)
        _b = np.zeros(0, dtype=ctypes.c_int)

        _hs = np.zeros(1, dtype=ctypes.c_int)
        _h = np.zeros(0, dtype=ctypes.c_int)

        _s = np.zeros(0, dtype=ctypes.c_double)

        _r = np.zeros(0, dtype=ctypes.c_int)

        for hx, bhx in enumerate(_cell_pairs):

            # Boundary and Halo start index.
            _bs = np.append(_bs, ctypes.c_int(len(bhx[0])))
            _hs = np.append(_hs, ctypes.c_int(len(bhx[1])))

            # Actual cell indices
            _b = np.append(_b, bhx[0])
            _h = np.append(_h, bhx[1])

            # Offset shifts for periodic boundary
            _s = np.append(_s, bhx[2])

            # reverse lookup required for cuda.
            _r = np.append(_r, np.array(hx * np.ones(len(bhx[0])), dtype=ctypes.c_int))

        self._boundary_groups_start_end_indices = cuda_base.Array(_bs, dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = cuda_base.Array(_hs, dtype=ctypes.c_int)

        # print "CA =", self.occ_matrix.domain.cell_array
        # print _b

        self._boundary_cell_groups = cuda_base.Array(_b, dtype=ctypes.c_int)
        self._halo_cell_groups = cuda_base.Array(_h, dtype=ctypes.c_int)


        # print "SHIFTS"
        self._halo_shifts = cuda_base.Array(_s, dtype=ctypes.c_double)
        # print "E_SHIFTS", self._halo_shifts.ctypes_data

        self._reverse_lookup = cuda_base.Array(_r, dtype=ctypes.c_int)

        self._version = self._occ_matrix.domain.cell_array.version



    @property
    def get_boundary_cell_groups(self):
        """
        Get the local boundary cells to pack for each halo. Formatted as an cuda_base.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting points within the first array.
        """

        #assert self._host_halo_handle is not None, "No host halo setup."

        #_t = self._host_halo_handle.get_boundary_cell_groups

        #self._boundary_cell_groups.sync_from_version(_t[0])
        #self._boundary_groups_start_end_indices.sync_from_version(_t[1])

        if self._version < self._occ_matrix.domain.cell_array.version:
            self._get_pairs()

        return self._boundary_cell_groups, self._boundary_groups_start_end_indices

    @property
    def get_halo_cell_groups(self):
        """
        Get the local halo cells to unpack into for each halo. Formatted as an cuda_base.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local halo cell indices to unpack into, array of starting points within the first array.
        """
        # assert self._host_halo_handle is not None, "No host halo setup."

        #_t = self._host_halo_handle.get_halo_cell_groups

        #self._halo_cell_groups.sync_from_version(_t[0])
        #self._halo_groups_start_end_indices.sync_from_version(_t[1])

        if self._version < self._occ_matrix.domain.cell_array.version:
            self._get_pairs()

        return self._halo_cell_groups, self._halo_groups_start_end_indices

    @property
    def get_boundary_cell_contents_count(self):
        """
        Get the number of particles in the corresponding cells for each halo. These are needed such that
        the cell list can be created without inspecting the positions of recvd particles.

        :return: Tuple: Cell contents count for each cell in same order as local boundary cell list, Exchange sizes for each halo.
        """
        if not self._init:
            print("cuda_halo.CartesianHalo error. Library not initalised,"
                  " this error means the internal setup failed.")
            abort()

        # TODO: run sizes calculation here
        self._exchange_sizes.zero()


        return self._boundary_groups_contents_array, self._exchange_sizes

    @property
    def get_position_shifts(self):

        '''Calculate flag to determine if a boundary between processes is also a boundary in domain.'''

        if self._version < self._occ_matrix.domain.cell_array.version:
            self._get_pairs()

        return self._halo_shifts

    @property
    def get_boundary_cell_to_halo_map(self):

        if self._version < self._occ_matrix.domain.cell_array.version:
            self._get_pairs()

        return self._reverse_lookup



def copy_h2d_exclusive_scan(in_array, out_array):
    """
    Copy an Array and compute an exclusive scan on the copy. Resizes out array
    to length of in array plus 1.
    :param in_array:
    :param out_array:
    """

    assert type(in_array) is cuda_base.Array, "in_array as incorrect type"
    assert type(out_array) is cuda_base.Array, "out_array as incorrect type"

    if out_array.ncomp != (in_array.ncomp + 1):
        out_array.realloc(in_array.ncomp + 1)

    cuda_runtime.cuda_mem_cpy(d_ptr=out_array.ctypes_data,
                              s_ptr=in_array.ctypes_data,
                              size=in_array.ncomp * ctypes.sizeof(in_array.dtype),
                              cpy_type="cudaMemcpyHostToDevice")

    cuda_runtime.cuda_exclusive_scan(out_array, in_array.ncomp+1)

    return


'''
    const int length,
    const int max_count,
    const int occ_matrix_stride,
    const int n_local,
    const int* __restrict__ d_halo_indices,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_halo_scan,
    int* __restrict__ d_occ_matrix
'''


def update_cell_occ_matrix(
    length,
    max_count,
    occ_matrix_stride,
    n_local,
    d_halo_indices,
    d_ccc,
    d_halo_scan,
    d_occ_matrix
    ):

    #print "occ halo pointer pre", d_occ_matrix.ctypes_data
    cuda_runtime.cuda_err_check(
    cuda_mpi.LIB_CUDA_MPI['cudaHaloFillOccupancyMatrix'](
        ctypes.c_int32(length),
        ctypes.c_int32(max_count),
        ctypes.c_int32(occ_matrix_stride),
        ctypes.c_int32(n_local),
        d_halo_indices.ctypes_data,
        d_ccc.ctypes_data,
        d_halo_scan.ctypes_data,
        d_occ_matrix.ctypes_data
    )
    )

    #print "occ halo pointer post", d_occ_matrix.ctypes_data



'''
const int * __restrict__ h_b_arr,
const int * __restrict__ d_b_scan,
int * __restrict__ h_p_count
'''


def update_send_counts(
        host_b_se_indices,
        device_b_scan,
        host_send_counts):

    cuda_runtime.cuda_err_check(
    cuda_mpi.LIB_CUDA_MPI['cudaCopySendCounts'](
        host_b_se_indices.ctypes_data,
        device_b_scan.ctypes_data,
        host_send_counts.ctypes_data
    ))























