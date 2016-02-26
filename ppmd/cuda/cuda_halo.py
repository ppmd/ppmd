"""
Methods to aid CUDA halo exchanges.
"""

# system level imports
import ctypes
import numpy as np

# package level imports
import ppmd.mpi as mpi

# cuda level imports
import cuda_runtime
import cuda_build
import cuda_cell
import cuda_base

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

    xr = range(1, cell_array[0] - 1)[slicexyz[0]]
    yr = range(1, cell_array[1] - 1)[slicexyz[1]]
    zr = range(1, cell_array[2] - 1)[slicexyz[2]]

    if type(xr) is not list:
        xr = [xr]
    if type(yr) is not list:
        yr = [yr]
    if type(zr) is not list:
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
        if mpi.MPI_HANDLE.top[ix] == 0 and mpi.MPI_HANDLE.periods[ix] == 1 and direction[ix] == -1:
            shift[ix] = extent[ix]
        if mpi.MPI_HANDLE.top[ix] == mpi.MPI_HANDLE.dims[ix] - 1 and mpi.MPI_HANDLE.periods[ix] == 1 and direction[ix] == 1:
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

        # self._setup()

    @property
    def occ_matrix(self):
        """
        Returns the occupancy matrix involved.
        :return:
        """
        return self._occ_matrix



    def _setup(self):
        """
        Internally setup the libraries for the calculation of exchange sizes.
        """

        p1_args = '''
                  int max_layers,
                  cuda_Matrix<int> COM, // cell occupancy matrix
                  cuda_Array<int> CCC,  // cell contents count
                  cuda_Array<int> BCG,  // Boundary cell groups
                  cuda_Array<int> ES,   // Exchange sizes
                  cuda_Array<int> BGCA  // Boundary groups cell arrays (sizes for each cell)
                  '''

        _p1_header_code = '''
        //Header
        #include <cuda_generic.h>
        extern "C" int CartesianHaloL0_0(%(ARGS)s);
        ''' %{'ARGS': p1_args}

        _p1_code = '''
        //source

        int CartesianHaloL0_0(%(ARGS)s){
            int err = 0;

            err = *(BCG.ncomp);

            return err;
        }
        ''' % {'ARGS':p1_args}

        #self._p1_lib = cuda_build.simple_lib_creator(_p1_header_code, _p1_code, 'CartesianHaloL0')





        # RUNNING

        '''
        self._boundary_cell_groups.sync_from_version(halo.HALOS.get_boundary_cell_groups()[0])

        if self._boundary_groups_contents_array.ncomp < self._boundary_cell_groups.ncomp:
            self._boundary_groups_contents_array.realloc(self._boundary_cell_groups.ncomp)

        self._exchange_sizes.zero()






        args = [
                ctypes.c_int(self._occ_matrix.layers_per_cell),
                self._occ_matrix.matrix.struct,
                self._occ_matrix.cell_contents_count.struct,
                self._boundary_cell_groups.struct,
                self._exchange_sizes.struct,
                self._boundary_groups_contents_array.struct
                ]

        #print self._p1_lib['CartesianHaloL0_0'](*args)
        '''











        self._init = True


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
            print "cuda_halo.CartesianHalo error. Library not initalised, this error means the internal" \
                  "setup failed."
            quit()

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




HALOS = None






