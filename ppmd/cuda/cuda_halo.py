"""
Methods to aid CUDA halo exchanges.
"""

# system level imports
import ctypes

# package level imports
from ppmd import halo


# cuda level imports
import cuda_runtime
import cuda_build
import cuda_cell
import cuda_base

class CartesianHalo(object):

    def __init__(self, host_halo=halo.HALOS, occ_matrix=cuda_cell.OCCUPANCY_MATRIX):
        self._host_halo_handle = host_halo
        self._occ_matrix = occ_matrix
        self._init = False

        # vars init
        self._boundary_cell_groups = cuda_base.Array(dtype=ctypes.c_int)
        self._boundary_groups_start_end_indices = cuda_base.Array(ncomp=27, dtype=ctypes.c_int)
        self._halo_cell_groups = cuda_base.Array(dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = cuda_base.Array(ncomp=27, dtype=ctypes.c_int)
        self._boundary_groups_contents_array = cuda_base.Array(dtype=ctypes.c_int)
        self._exchange_sizes = cuda_base.Array(ncomp=26, dtype=ctypes.c_int)

        # ensure first update
        self._boundary_cell_groups.inc_version(-1)
        self._boundary_groups_start_end_indices.inc_version(-1)
        self._halo_cell_groups.inc_version(-1)
        self._halo_groups_start_end_indices.inc_version(-1)
        self._boundary_groups_contents_array.inc_version(-1)
        self._exchange_sizes.inc_version(-1)

        self._setup()

    def _setup(self):
        """
        Internally setup the libraries for the calculation of exchange sizes.
        """














        self._init = True


    @property
    def get_boundary_cell_groups(self):
        """
        Get the local boundary cells to pack for each halo. Formatted as an cuda_base.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting points within the first array.
        """

        assert self._host_halo_handle is not None, "No host halo setup."

        _t = self._host_halo_handle.get_boundary_cell_groups

        self._boundary_cell_groups.sync_from_version(_t[0])
        self._boundary_groups_start_end_indices.sync_from_version(_t[1])

        return self._boundary_cell_groups, self._boundary_groups_start_end_indices

    @property
    def get_halo_cell_groups(self):
        """
        Get the local halo cells to unpack into for each halo. Formatted as an cuda_base.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local halo cell indices to unpack into, array of starting points within the first array.
        """
        assert self._host_halo_handle is not None, "No host halo setup."

        _t = self._host_halo_handle.get_halo_cell_groups

        self._halo_cell_groups.sync_from_version(_t[0])
        self._halo_groups_start_end_indices.sync_from_version(_t[1])

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











