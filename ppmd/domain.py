from __future__ import print_function, division, absolute_import

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import numpy as np
import math
import ctypes
import os

# package level
from ppmd import data, host,  mpi, runtime, pio, opt
from ppmd.lib import build

_LIB_SOURCES = os.path.join(os.path.dirname(__file__), 'lib/domain/')


##############################################################################################################
# BASE DOMAIN HALO
##############################################################################################################


class BaseDomainHalo(object):
    """
    A cell based domain for mpi/private memory. Creates a shell of halos cells around
    each processes internal cells as halos.

    """

    def __init__(self, extent=None, periods=(1, 1, 1), comm=mpi.MPI.COMM_WORLD):
        self._init_cells = False
        self._init_decomp = False

        self.version_id = 0
        self._periods = periods

        self._extent = data.ScalarArray(ncomp=3, dtype=ctypes.c_double)
        self._extent_global = data.ScalarArray(ncomp=3, dtype=ctypes.c_double)
        self._boundary_outer = None
        self._boundary = None

        self._init_extent = False
        if extent is not None:
            self.set_extent(extent)


        self._cell_array = data.ScalarArray(np.array([1, 1, 1]), dtype=ctypes.c_int)
        self._cell_edge_lengths = data.ScalarArray(np.array([1., 1., 1.], dtype=ctypes.c_double))

        self._halos = True


        self.boundary_condition = None

        #vars to return boudary cells
        self._boundary_cell_version = -1
        self._boundary_cells = None

        self.comm = None
        self._parent_comm = comm

    @property
    def dims(self):
        return mpi.cartcomm_dims_xyz(self.comm)


    def get_boundary_cells(self):
        """
        Return a host.Array containing the boundary cell indices of the domain.
        """

        if self._boundary_cell_version < self._cell_array.version:
            _ca = self._cell_array
            _count = (_ca[0] - 2) * (_ca[1] - 2) * (_ca[2] - 2) - (_ca[0] - 4) * (_ca[1] - 4) * (_ca[2] - 4)

            self._boundary_cells = host.Array(ncomp=_count, dtype=ctypes.c_int)
            m = 0

            for ix in range(1, _ca[0] - 1):
                for iy in range(1, _ca[1] - 1):

                    self._boundary_cells[m] = ix + _ca[0]*(iy + _ca[1])
                    self._boundary_cells[m + (_ca[0]-2) * (_ca[1]-2) ] = ix + _ca[0]*(iy + (_ca[2] - 2)*_ca[1])
                    m += 1
            m += (_ca[0]-2)*(_ca[1]-2)

            for ix in range(1, _ca[0] - 1):
                for iz in range(2, _ca[2] - 2):
                        self._boundary_cells[m] = ix + _ca[0]*(1 + iz*_ca[1])
                        self._boundary_cells[m + (_ca[0]-2) * (_ca[2]-4) ] = ix + _ca[0]*((_ca[1] - 2) + iz*_ca[1])
                        m += 1

            m += (_ca[0]-2)*(_ca[2]-4)

            for iy in range(2, _ca[1] - 2):
                for iz in range(2, _ca[2] - 2):
                        self._boundary_cells[m] = 1 + _ca[0]*(iy + iz*_ca[1])
                        self._boundary_cells[m + (_ca[1]-4) * (_ca[2]-4) ] = _ca[0]-2 + _ca[0]*(iy + iz*_ca[1])
                        m += 1

            m += (_ca[1]-4)*(_ca[2]-4)
            

            self._boundary_cell_version = self._cell_array.version


        return self._boundary_cells

    @property
    def cell_array(self):
        """
        Return cell array.
        """
        return self._cell_array

    @property
    def halos(self):
        return self._halos

    def set_extent(self, new_extent=np.array([1., 1., 1.])):
        """
        Set domain extents
        
        :arg np.array(3,1) new_extent: New extents.
        
        """
        self._extent[0:4] = new_extent
        self._extent_global[0:4] = new_extent

        self._boundary_outer = (
            -0.5 * self._extent[0], 0.5 * self._extent[0],
            -0.5 * self._extent[1], 0.5 * self._extent[1],
            -0.5 * self._extent[2], 0.5 * self._extent[2]
        )

        self._boundary = (
            -0.5 * self._extent[0], 0.5 * self._extent[0],
            -0.5 * self._extent[1], 0.5 * self._extent[1],
            -0.5 * self._extent[2], 0.5 * self._extent[2]
        )

        if self._init_decomp:
            print("WARNING EXTENT CHANGED AFTER DECOMP")

            self._distribute_domain()

        self._init_extent = True
        self.version_id += 1


    def mpi_decompose(self, mpi_grid=None):
        if self._init_decomp:
            return True
            # print("WARNING: domain already spatially decomposed")
        
        mpisize = self._parent_comm.size

        if mpi_grid is None:
            if self._init_extent:
                sf = min(float(self.extent[0]),
                         float(self.extent[1]),
                         float(self.extent[2]))
                sc = (int(self.extent[0]/sf)*1000,
                      int(self.extent[1]/sf)*1000,
                      int(self.extent[2]/sf)*1000)

                _dims = _find_domain_decomp(sc, mpisize)
            else:
                _dims = _find_domain_decomp_no_extent(mpisize)

        else:
            assert mpi_grid[0]*mpi_grid[1]*mpi_grid[2]==mpisize,\
                "Incompatible MPI rank structure"
            _dims = mpi_grid

         # Create cartesian communicator
        _dims = tuple(_dims)
        
        self.comm = self._parent_comm.Create_cart(
            _dims[::-1], 
            (bool(self._periods[2]), bool(self._periods[1]), bool(self._periods[0])),
            True
        )

        self._init_decomp = True

        if self._init_extent:
            self._distribute_domain()

        self.version_id += 1
        return True

    def _distribute_domain(self):

        _top = mpi.cartcomm_top_xyz(self.comm)
        _dims = mpi.cartcomm_dims_xyz(self.comm)

        opt.PROFILE[self.__class__.__name__+':mpi_dims'] = (_dims)

        self._extent[0] = self._extent_global[0] / _dims[0]
        self._extent[1] = self._extent_global[1] / _dims[1]
        self._extent[2] = self._extent_global[2] / _dims[2]

        _boundary = (
            -0.5 * self._extent_global[0] + _top[0] * self._extent[0],
            -0.5 * self._extent_global[0] + (_top[0] + 1.) * self._extent[0],
            -0.5 * self._extent_global[1] + _top[1] * self._extent[1],
            -0.5 * self._extent_global[1] + (_top[1] + 1.) * self._extent[1],
            -0.5 * self._extent_global[2] + _top[2] * self._extent[2],
            -0.5 * self._extent_global[2] + (_top[2] + 1.) * self._extent[2]
        )

        self._boundary = data.ScalarArray(_boundary, dtype=ctypes.c_double)
        self._boundary_outer = data.ScalarArray(_boundary, dtype=ctypes.c_double)


    def cell_decompose(self, cell_width=None):

        assert cell_width is not None, "ERROR: No cell size passed."
        assert cell_width > 10.**-14, "ERROR: requested cell size stupidly small."

        if not self._init_decomp:
            print("WARNING: domain not spatial decomposed, see mpi_decompose()")

        cell_width = float(cell_width)

        self._cell_array[0] = int(self._extent[0] / cell_width)
        self._cell_array[1] = int(self._extent[1] / cell_width)
        self._cell_array[2] = int(self._extent[2] / cell_width)

        assert self._cell_array[0] > 0, "Too many MPI ranks in dir 0"
        assert self._cell_array[1] > 0, "Too many MPI ranks in dir 1"
        assert self._cell_array[2] > 0, "Too many MPI ranks in dir 2"

        self._cell_edge_lengths[0] = self._extent[0] / self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1] / self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2] / self._cell_array[2]

        self._cell_array[0] += 2
        self._cell_array[1] += 2
        self._cell_array[2] += 2


        _boundary = (
            self._boundary[0] - self._cell_edge_lengths[0],
            self._boundary[1] + self._cell_edge_lengths[0],
            self._boundary[2] - self._cell_edge_lengths[1],
            self._boundary[3] + self._cell_edge_lengths[1],
            self._boundary[4] - self._cell_edge_lengths[2],
            self._boundary[5] + self._cell_edge_lengths[2]
        )

        self._boundary_outer = data.ScalarArray(_boundary, dtype=ctypes.c_double)

        self._init_cells = True

        opt.PROFILE[self.__class__.__name__+':cell_array'] = (self.cell_array[:])

        self.version_id += 1
        return True


    @property
    def volume(self):
        """
        Return domain volume.
        """
        return self._extent[0] * self._extent[1] * self._extent[2]

    @property
    def boundary(self):
        """
        Return local domain boundary
        """
        return self._boundary

    @property
    def boundary_outer(self):
        """
        Return local domain boundary
        """
        if not self._init_cells and runtime.VERBOSE > 0:
            print("WARNING: No cell decomposition, outer boundary same as inner")

        return self._boundary_outer

    @property
    def extent(self):
        """
        Returns list of domain extents.
        """
        # return self._extent_outer
        return self._extent_global

    @property
    def extent_internal(self):
        """
        Returns list of domain extents.
        """

        return self._extent


    def get_shift(self):

        _sfd = host.Array(ncomp=26*3, dtype=ctypes.c_double)

        dims = mpi.cartcomm_dims_xyz(self.comm)
        top = mpi.cartcomm_top_xyz(self.comm)
        periods = mpi.cartcomm_periods_xyz(self.comm)

        for dx in range(26):
            dir = mpi.recv_modifiers[dx]

            for ix in range(3):

                if top[ix] == 0 and \
                   periods[ix] == 1 and \
                   dir[ix] == -1:

                    _sfd[dx*3 + ix] = self.extent[ix]

                elif top[ix] == dims[ix] - 1 and \
                   periods[ix] == 1 and \
                   dir[ix] == 1:
                    _sfd[dx*3 + ix] = -1. * self.extent[ix]
                else:
                    _sfd[dx*3 + ix] = 0.0
        return _sfd

    @property
    def cell_edge_lengths(self):
        """
        Return cell edge lengths.
        """
        return self._cell_edge_lengths

    @property
    def cell_count(self):
        """
        Return cell count for domain.
        """
        return self._cell_array[0] * self._cell_array[1] * self._cell_array[2]

def factor(n):
    return [ix for ix in range(1, n // 2 + 1) if not n % ix] + [n]

def pfactor(n):
    lst = []
    l = 2
    while l <= n:
        if n % l == 0:
            n //= l
            lst.append(l)
        else:
            l += 1
    return lst


def get_domain_decomp(nproc):
    return _find_domain_decomp_no_extent(nproc)


def _find_domain_decomp_no_extent(nproc):
    """
    find a decomp
    :param nproc:
    :return:
    """
    assert nproc is not None, "No number of processes passed"


    if runtime.MPI_DIMS is not None:
        assert len(runtime.MPI_DIMS) == 3, "bad number of mpi dims defined"
        p = runtime.MPI_DIMS[0] * runtime.MPI_DIMS[1] * runtime.MPI_DIMS[2]
        assert p == nproc, "bad number of dims predefined {}, {}".format(p, nproc)
        return runtime.MPI_DIMS


    return mpi.MPI.Compute_dims(nproc, 3)


def _find_domain_decomp(global_cell_array=None, nproc=None):
    """
    find a decomp
    :param global_cell_array:
    :param nproc:
    :return:
    """

    '''Order domain dimension sizes in descending order'''
    _cal = [[0, global_cell_array[0]], [1, global_cell_array[1]], [2, global_cell_array[2]]]
    _cal.sort(key=lambda x: x[1], reverse=True)

    '''Order processor calculated dimension sizes in descending order'''
    _NP = list(_find_domain_decomp_no_extent(nproc))
    _NP.sort(reverse=True)

    '''Try to match avaible processor dimensions to phyiscal cells'''

    success = True

    _dims = [0, 0, 0]
    for i in range(3):
        ix = _cal[i][0]
        if _cal[i][1] < _NP[i]:
            print("ERROR matching domain to processes, dimension %(DIM)s" \
                  %{'DIM': str(ix)})
            success = False

        _dims[ix] = _NP[i]

    if not success:
        raise RuntimeError("Processor grid error, suitable layout search failed." + str(_dims[:]) + str(global_cell_array[:]))

    return _dims


def _get_cell_distribution(global_cell_array=None, dims=None, top=None):

    # blocks per cell
    _bsc = [int(math.ceil( float(global_cell_array[0]) / float(dims[0]))),
            int(math.ceil( float(global_cell_array[1]) / float(dims[1]))),
            int(math.ceil( float(global_cell_array[2]) / float(dims[2])))]

    # print dims
    # print _bsc

    # Calculate global distribution of cells
    _bs = []
    for ix in range(3):
        _tmp = []

        if (_bsc[ix]*(dims[ix]-1)) < global_cell_array[ix]:

            for iy in range(dims[ix] - 1):
                _tmp.append(int(_bsc[ix]))
            _tmp.append(int(global_cell_array[0] - (dims[ix] - 1) * _bsc[ix]))

        else:

            R = global_cell_array[ix] % dims[ix]
            for iy in range(R):
                _tmp.append(_bsc[ix])
            for iy in range(dims[ix] - R):
                _tmp.append((global_cell_array[ix] - R * _bsc[ix]) // (dims[ix] - R) )

        assert len(_tmp) == dims[ix], "DD size missmatch, dim: " + str(ix) + " " + str(_tmp[:])
        _tsum = 0
        for tx in _tmp:
            _tsum += tx

        assert _tsum == global_cell_array[ix], "DD failure to assign cells, dim: " + str(ix) + " " + str(_tmp[:])

        _bs.append(_tmp)

    if runtime.VERBOSE > 1:
        pio.pprint("Cell layout", _bs)

    # Get local cell array
    local_cell_array = (_bs[0][top[0]], _bs[1][top[1]], _bs[2][top[2]])

    return local_cell_array, _bs


class BoundaryTypePeriodic(object):
    """
    Class to hold and perform periodic boundary conditions.

    :arg state_in: State on which to apply periodic boundaries to.
    """

    def __init__(self, state_in=None):
        self.state = state_in

        # Initialise timers
        self.timer_apply = ppmd.opt.Timer(runtime.TIMER, 0)
        self.timer_search = ppmd.opt.Timer(runtime.TIMER, 0)
        self.timer_move = ppmd.opt.Timer(runtime.TIMER, 0)

        # One proc PBC lib
        self._one_process_pbc_lib = None
        # Escape guard lib
        self._escape_guard_lib = None
        self._escape_count = None
        self._escape_linked_list = None
        self._flag = host.Array(ncomp=1, dtype=ctypes.c_int)

    def set_state(self, state_in=None):
        assert state_in is not None, "BoundaryTypePeriodic error: No state passed."
        self._escape_guard_lib = None
        self._one_process_pbc_lib = None
        self._escape_linked_list = None
        self.state = state_in

    def apply(self):
        """
        Enforce the boundary conditions on the held state.
        """

        comm = self.state.domain.comm
        self.timer_apply.start()

        self._flag.data[0] = 0

        if comm.Get_size() == 1:
            if self._one_process_pbc_lib is None:
                self._init_one_proc_lib()
            self._one_process_pbc_lib(
                ctypes.c_int(self.state.npart_local),
                self.state.get_position_dat().ctypes_data,
                self.state.domain.extent.ctypes_data,
                self._flag.ctypes_data
            )

        else:

            if self._escape_guard_lib is None:
                self._init_escape_lib()

            # reset linked list
            self._escape_linked_list[0:26:] = -1
            self._escape_count[::] = 0

            num_slots = 26 + 2 * self.state.npart_local
            if self._escape_linked_list.ncomp < num_slots:
                # alloc slightly more than needed to avoid reallocs each
                #  iteration
                self._escape_linked_list.realloc(num_slots+16)

            self._escape_guard_lib(
                ctypes.c_int(self.state.npart_local),
                self._escape_count.ctypes_data,
                self._bin_to_lin.ctypes_data,
                self._escape_linked_list.ctypes_data,
                self.state.domain.boundary.ctypes_data,
                self.state.get_position_dat().ctypes_data
            )

            self.timer_move.start()
            self.state.move_to_neighbour(self._escape_linked_list,
                                         self._escape_count,
                                         self.state.domain.get_shift())
            self.timer_move.pause()

        self.timer_apply.pause()

        return self._flag.data[0]


    def _init_escape_lib(self):
        ''' Create a lookup table between xor map and linear index for direction '''
        self._bin_to_lin = data.ScalarArray(ncomp=57, dtype=ctypes.c_int)
        _lin_to_bin = np.zeros(26, dtype=ctypes.c_int)

        '''linear to xor map'''
        _lin_to_bin[0] = 1 ^ 2 ^ 4
        _lin_to_bin[1] = 2 ^ 1
        _lin_to_bin[2] = 32 ^ 2 ^ 1
        _lin_to_bin[3] = 4 ^ 1
        _lin_to_bin[4] = 1
        _lin_to_bin[5] = 32 ^ 1
        _lin_to_bin[6] = 4 ^ 1 ^ 16
        _lin_to_bin[7] = 1 ^ 16
        _lin_to_bin[8] = 32 ^ 16 ^ 1

        _lin_to_bin[9] = 2 ^ 4
        _lin_to_bin[10] = 2
        _lin_to_bin[11] = 32 ^ 2
        _lin_to_bin[12] = 4
        _lin_to_bin[13] = 32
        _lin_to_bin[14] = 4 ^ 16
        _lin_to_bin[15] = 16
        _lin_to_bin[16] = 32 ^ 16

        _lin_to_bin[17] = 8 ^ 2 ^ 4
        _lin_to_bin[18] = 2 ^ 8
        _lin_to_bin[19] = 32 ^ 2 ^ 8
        _lin_to_bin[20] = 4 ^ 8
        _lin_to_bin[21] = 8
        _lin_to_bin[22] = 32 ^ 8
        _lin_to_bin[23] = 4 ^ 8 ^ 16
        _lin_to_bin[24] = 8 ^ 16
        _lin_to_bin[25] = 32 ^ 16 ^ 8

        '''inverse map, probably not ideal'''
        for ix in range(26):
            self._bin_to_lin[_lin_to_bin[ix]] = ix

        # Number of escaping particles in each direction
        self._escape_count = host.Array(np.zeros(26), dtype=ctypes.c_int)

        # Linked list to store the ids of escaping particles in a similar way
        # to the cell list.
        # | [0-25 escape directions, index of first in direction] [26-end
        # current id and index of next id, (id, next_index) ]|

        self._escape_linked_list = host.Array(-1 * np.ones(26 + 2 * self.state.npart_local), dtype=ctypes.c_int)

        dtype = self.state.get_position_dat().dtype
        assert self.state.domain.boundary.dtype == dtype

        self._escape_guard_lib = ppmd.lib.build.lib_from_file_source(
            _LIB_SOURCES + 'EscapeGuard',
            'EscapeGuard',
            {
                'SUB_REAL': self.state.get_position_dat().ctype,
                'SUB_INT': self._bin_to_lin.ctype
            }
        )['EscapeGuard']

    def _init_one_proc_lib(self):

        assert self.state.domain.extent.dtype == self.state.get_position_dat().dtype
        self._one_process_pbc_lib = build.lib_from_file_source(
            _LIB_SOURCES + 'OneRankPBC',
            'OneRankPBC',
            {
                'SUB_REAL': self.state.get_position_dat().ctype,
                'SUB_INT': self._flag.ctype
            }
        )['OneRankPBC']












