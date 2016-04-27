
# system level
import ctypes
import numpy as np

import cell
import kernel
import build
import runtime
import mpi
import host

################################################################################################################
# HALO DEFINITIONS
################################################################################################################
from ppmd import pio


class CartesianHalo(object):
    """
    Class to contain and control cartesian halo transfers.
    """

    def __init__(self):

        self.timer = runtime.Timer(runtime.TIMER, 0, start=True)

        # List that tracks the local cell stucture to detect changes.
        self._ca_copy = [None, None, None]
        self._halo_setup_prepare()

        self.timer.stop("halo setup time.")



    def _halo_setup_prepare(self):

        """Determine sources and destinations"""
        _recv_modifiers = [
            [-1, -1, -1],  # 0
            [0, -1, -1],  # 1
            [1, -1, -1],  # 2
            [-1, 0, -1],  # 3
            [0, 0, -1],  # 4
            [1, 0, -1],  # 5
            [-1, 1, -1],  # 6
            [0, 1, -1],  # 7
            [1, 1, -1],  # 8

            [-1, -1, 0],  # 9
            [0, -1, 0],  # 10
            [1, -1, 0],  # 11
            [-1, 0, 0],  # 12
            [1, 0, 0],  # 13
            [-1, 1, 0],  # 14
            [0, 1, 0],  # 15
            [1, 1, 0],  # 16

            [-1, -1, 1],  # 17
            [0, -1, 1],  # 18
            [1, -1, 1],  # 19
            [-1, 0, 1],  # 20
            [0, 0, 1],  # 21
            [1, 0, 1],  # 22
            [-1, 1, 1],  # 23
            [0, 1, 1],  # 24
            [1, 1, 1],  # 25
        ]

        # '''Count how many halos out of 26 exist. Some may not if directions are not periodic'''
        # _num_halos = 0
        self._send_list = []
        self._recv_list = []
        for ix in _recv_modifiers:
            _tr = mpi.MPI_HANDLE.shift([-1 * ix[0], -1 * ix[1], -1 * ix[2]])
            self._send_list.append(_tr)

            _tr = mpi.MPI_HANDLE.shift(ix)
            self._recv_list.append(_tr)

        '''Array to store the number of particles to exchange for each halo'''
        self._exchange_sizes = host.Array(ncomp=26, dtype=ctypes.c_int)

        # CELL INDICES TO PACK (local boundary) =======================

        _ca = cell.cell_list.domain.cell_array

        _E = _ca[0] * _ca[1] * (_ca[2] - 1) - _ca[0] - 1  # End
        _TS = _E - _ca[0] * (_ca[1] - 2) + 2  # Top Start

        _BE = _ca[0] * (2 * _ca[1] - 1) - 1  # Bottom end
        _BS = _ca[0] * (_ca[1] + 1) + 1  # Bottom start

        _tmp4 = []
        for ix in range(_ca[1] - 2):
            _tmp4 += range(_TS + ix * _ca[0], _TS + (ix + 1) * _ca[0] - 2, 1)

        _tmp10 = []
        for ix in range(_ca[2] - 2):
            _tmp10 += range(_BE - _ca[0] + 2 + ix * _ca[0] * _ca[1],
                            _BE + ix * _ca[0] * _ca[1], 1)

        _tmp12 = []
        for ix in range(_ca[2] - 2):
            _tmp12 += range(_BS + _ca[0] - 3 + ix * _ca[0] * _ca[1],
                            _BE + ix * _ca[0] * _ca[1], _ca[0])

        _tmp13 = []
        for ix in range(_ca[2] - 2):
            _tmp13 += range(_BS + ix * _ca[0] * _ca[1], _BE + ix * _ca[0] * _ca[1], _ca[0])

        _tmp15 = []
        for ix in range(_ca[2] - 2):
            _tmp15 += range(_BS + ix * _ca[0] * _ca[1],
                            _BS + _ca[0] - 2 + ix * _ca[0] * _ca[1], 1)

        _tmp21 = []
        for ix in range(_ca[1] - 2):
            _tmp21 += range(_BS + ix * _ca[0], _BS + (ix + 1) * _ca[0] - 2, 1)

        self._boundary_cell_indices = [
            [_E - 1],
            range(_E - _ca[0] + 2, _E, 1),
            [_E - _ca[0] + 2],
            range(_TS + _ca[0] - 3, _E, _ca[0]),
            _tmp4,
            range(_TS, _E, _ca[0]),
            [_TS + _ca[0] - 3],
            range(_TS, _TS + _ca[0] - 2, 1),
            [_TS],

            range(_BE - 1, _E, _ca[0] * _ca[1]),
            _tmp10,
            range(_BE - _ca[0] + 2, _E, _ca[0] * _ca[1]),
            _tmp12,
            _tmp13,
            range(_BS + _ca[0] - 3, _E, _ca[0] * _ca[1]),
            _tmp15,
            range(_BS, _E, _ca[0] * _ca[1]),

            [_BE - 1],
            range(_BE - _ca[0] + 2, _BE, 1),
            [_BE - _ca[0] + 2],
            range(_BS + _ca[0] - 3, _BE, _ca[0]),
            _tmp21,
            range(_BS, _BE, _ca[0]),
            [_BS + _ca[0] - 3],
            range(_BS, _BS + _ca[0] - 2, 1),
            [_BS]
        ]

        # =============================================================
        # LOCAL CELL INDICES TO SORT INTO (local halo)

        _LE = _ca[0] * _ca[1] * _ca[2]
        _LTS = _LE - _ca[0] * _ca[1]

        _LBE = _ca[0] * _ca[1]
        _LBS = 0

        _Ltmp4 = []
        for ix in range(_ca[1] - 2):
            _Ltmp4 += range(_LBS + 1 + (ix + 1) * _ca[0], _LBS + (ix + 2) * _ca[0] - 1, 1)

        _Ltmp10 = []
        for ix in range(_ca[2] - 2):
            _Ltmp10 += range(_LBS + 1 + (ix + 1) * _ca[0] * _ca[1],
                             _LBS + (ix + 1) * _ca[0] * _ca[1] + _ca[0] - 1, 1)

        _Ltmp12 = []
        for ix in range(_ca[2] - 2):
            _Ltmp12 += range(_LBS + _ca[0] + (ix + 1) * _ca[0] * _ca[1],
                             _LBS + _ca[0] * (_ca[1] - 1) + (ix + 1) * _ca[0] * _ca[1], _ca[0])

        _Ltmp13 = []
        for ix in range(_ca[2] - 2):
            _Ltmp13 += range(_LBS + 2 * _ca[0] - 1 + (ix + 1) * _ca[0] * _ca[1],
                             _LBE + (ix + 1) * _ca[0] * _ca[1] - 3, _ca[0])

        _Ltmp15 = []
        for ix in range(_ca[2] - 2):
            _Ltmp15 += range(_LBE + (ix + 1) * _ca[0] * _ca[1] - _ca[0] + 1,
                             _LBE + (ix + 1) * _ca[0] * _ca[1] - 1, 1)

        _Ltmp21 = []
        for ix in range(_ca[1] - 2):
            _Ltmp21 += range(_LTS + (ix + 1) * _ca[0] + 1, _LTS + (ix + 2) * _ca[0] - 1, 1)

        self._halo_cell_indices = [
            [_LBS],
            range(_LBS + 1, _LBS + _ca[0] - 1, 1),
            [_LBS + _ca[0] - 1],
            range(_LBS + _ca[0], _LBS + _ca[0] * (_ca[1] - 2) + 1, _ca[0]),
            _Ltmp4,
            range(_LBS + 2 * _ca[0] - 1, _LBE - _ca[0], _ca[0]),
            [_LBE - _ca[0]],
            range(_LBE - _ca[0] + 1, _LBE - 1, 1),
            [_LBE - 1],

            range(_LBS + _ca[0] * _ca[1], _LBS + (_ca[2] - 2) * _ca[0] * _ca[1] + 1,
                  _ca[0] * _ca[1]),
            _Ltmp10,
            range(_LBE + _ca[0] - 1, _LBE + (_ca[2] - 2) * _ca[0] * _ca[1] + _ca[0] - 1,
                  _ca[0] * _ca[1]),
            _Ltmp12,
            _Ltmp13,
            range(_LBE + _ca[0] * (_ca[1] - 1), _LE - _ca[0] * _ca[1], _ca[0] * _ca[1]),
            _Ltmp15,
            range(_LBE + _ca[0] * _ca[1] - 1, _LE - _ca[0] * _ca[1], _ca[0] * _ca[1]),

            [_LTS],
            range(_LTS + 1, _LTS + _ca[0] - 1, 1),
            [_LTS + _ca[0] - 1],
            range(_LTS + _ca[0], _LTS + _ca[0] * (_ca[1] - 2) + 1, _ca[0]),
            _Ltmp21,
            range(_LTS + 2 * _ca[0] - 1, _LE - _ca[0], _ca[0]),
            [_LE - _ca[0]],
            range(_LE - _ca[0] + 1, _LE - 1, 1),
            [_LE - 1],
        ]

        # =============================================================

        '''How many cells are in each halo (internal to pack)'''
        self._boundary_cell_grouping_lengths = host.Array(ncomp=26, dtype=ctypes.c_int)

        '''How many cells are in each halo (external to recv)'''
        self._halo_cell_grouping_lengths = host.Array(ncomp=26, dtype=ctypes.c_int)

        _tmp_list = []
        _tmp_list_local = []
        for ix in range(26):
            # determine internal cells to pack
            if self._send_list[ix] > -1:
                self._boundary_cell_grouping_lengths[ix] = len(self._boundary_cell_indices[ix])
                _tmp_list += self._boundary_cell_indices[ix]
            else:
                self._boundary_cell_grouping_lengths[ix] = 0

            # determine halo cells to unpack into
            if self._recv_list[ix] > -1:
                self._halo_cell_grouping_lengths[ix] = len(self._halo_cell_indices[ix])
                _tmp_list_local += self._halo_cell_indices[ix]
            else:
                self._halo_cell_grouping_lengths[ix] = 0


        # Local boundary cells that are to be packed.

        '''Array containing the internal cell indices'''
        self._boundary_cell_groups = host.Array(_tmp_list, dtype=ctypes.c_int)

        '''SEND: create list to extract start and end points for each halo from above lists.'''
        self._boundary_groups_start_end_indices = host.Array(ncomp=27, dtype=ctypes.c_int)

        '''create cell contents array for each halo that is being sent.'''
        self._boundary_groups_contents_array = host.Array(ncomp=self._boundary_cell_groups.ncomp, dtype=ctypes.c_int)


        # Local halo cells that are to be unpacked into.

        '''Array containing the halo cell indices'''
        self._halo_cell_groups = host.Array(_tmp_list_local, dtype=ctypes.c_int)

        '''RECV: create list to extract start and end points for each halo from above lists.'''
        self._halo_groups_start_end_indices = host.Array(ncomp=27, dtype=ctypes.c_int)


        '''Indices for sending (local boundary)'''
        _start_index = 0
        self._boundary_groups_start_end_indices[0] = 0
        for ix in range(26):
            _start_index += self._boundary_cell_grouping_lengths[ix]
            self._boundary_groups_start_end_indices[ix + 1] = _start_index

        '''Indices for receving (local halo)'''
        _start_index = 0
        self._halo_groups_start_end_indices[0] = 0
        for ix in range(26):
            _start_index += self._halo_cell_grouping_lengths[ix]
            self._halo_groups_start_end_indices[ix + 1] = _start_index


        # Code to calculate exchange sizes
        # =============================================================

        _exchange_sizes_code = '''
        int start_index = 0;
        
        //loop over the different halos
        for(int ix = 0; ix < 26; ix++){
            
            // reset count
            ES[ix] = 0;
            
            // loop over the local cells in each halo.
            for(int iy = 0; iy < CIL[ix]; iy++){
                
                // increment count using the cell count made when constructing the cell list
                ES[ix] += CCC[CI[start_index+iy]];
                CCA[start_index+iy]=CCC[CI[start_index+iy]];
            }
            start_index+=CIL[ix];
        }
        
        '''

        _args = {
            'CCC': host.NullIntArray,  # cell countents count, number of particles in each cell.
            'ES': host.NullIntArray,  # total number of particles to be exchanged for each halo.
            'CI': host.NullIntArray,  # array containing the local cells to pass over for each halo.
            'CIL': host.NullIntArray,
            # array holding how many cells are in each halo used to pass over the above array
            'CCA': host.NullIntArray,  # array containing the particle count for each cell in the same order as CI
        }

        _headers = ['stdio.h']
        _kernel = kernel.Kernel('ExchangeSizeCalc', _exchange_sizes_code, None, _headers, None, None)
        self._exchange_sizes_lib = build.SharedLib(_kernel, _args)

        # ==============================================================

        self._ca_copy = [cell.cell_list.domain.cell_array[0],
                         cell.cell_list.domain.cell_array[1],
                         cell.cell_list.domain.cell_array[2]]

        # boundary cells to halo lookup
        self._boundary_halo_lookup = host.Array(ncomp=1, dtype=ctypes.c_int)
        self._boundary_lookup_lib = None

    def _exchange_size_calc(self):

        _args = {
            'CCC': cell.cell_list.cell_contents_count,
            'ES': self._exchange_sizes,
            'CI': self._boundary_cell_groups,
            'CIL': self._boundary_cell_grouping_lengths,
            'CCA': self._boundary_groups_contents_array,
        }

        self._exchange_sizes_lib.execute(dat_dict=_args)

    def check_valid(self):
        """
        Check if current values are still correct.

        :return: bool
        """
        if cell.cell_list.domain.cell_array[0] != self._ca_copy[0] or cell.cell_list.domain.cell_array[1] != self._ca_copy[1] or cell.cell_list.domain.cell_array[2] != self._ca_copy[2]:
            return False
        else:
            return True


    def _create_reverse_boundary_lookup(self):

        if self._boundary_halo_lookup.ncomp < self._boundary_cell_groups:
            self._boundary_halo_lookup.realloc(self._boundary_cell_groups.ncomp)

        if self._boundary_lookup_lib is None:
            _name = 'boundary_lookup_lib'

            _args = '''const int * %(R)s CCA,
                       const int * %(R)s CCA_I,
                       int * %(R)s BHL
            ''' % {'R': build.TMPCC.restrict_keyword}

            _header = '''
            #include <generic.h>
                int %(NAME)s(%(ARGS)s);
            ''' % {'NAME': _name, 'ARGS':_args}

            _src = '''
                int %(NAME)s(%(ARGS)s){

                    for (int ih = 0; ih < 26; ih++){

                        const int start = CCA_I[ih];
                        const int end = CCA_I[ih];

                        for (int ic = start; ic < end; ic++){

                            BHL[ic] = ih;

                        }
                    }


                    return 0;
                }
            ''' % {'NAME': _name, 'ARGS':_args}

            self._boundary_lookup_lib = build.simple_lib_creator(_header, _src, _name)

        args = [self._boundary_cell_groups.ctypes_data, self._boundary_groups_start_end_indices.ctypes_data, self._boundary_halo_lookup.ctypes_data]

        self._boundary_lookup_lib['boundary_lookup_lib'](*args)

    def get_boundary_cell_groups(self, reverse_lookup=False):
        """
        Get the local boundary cells to pack for each halo. Formatted as an data.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting points within the first array.
        """

        _trigger = False
        if not self.check_valid():
            _trigger = True
            self._halo_setup_prepare()

        if reverse_lookup and(_trigger or self._boundary_halo_lookup is None):
            self._create_reverse_boundary_lookup()

        if reverse_lookup:
            return self._boundary_cell_groups, self._boundary_groups_start_end_indices, self._boundary_halo_lookup
        else:
            return self._boundary_cell_groups, self._boundary_groups_start_end_indices

    @property
    def get_halo_cell_groups(self):
        """
        Get the local halo cells to unpack into for each halo. Formatted as an data.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local halo cell indices to unpack into, array of starting points within the first array.
        """

        if not self.check_valid():
            self._halo_setup_prepare()
        return self._halo_cell_groups, self._halo_groups_start_end_indices

    @property
    def get_boundary_cell_contents_count(self):
        """
        Get the number of particles in the corresponding cells for each halo. These are needed such that
        the cell list can be created without inspecting the positions of recvd particles.

        :return: Tuple: Cell contents count for each cell in same order as local boundary cell list, Exchange sizes for each halo.
        """
        if not self.check_valid():
            self._halo_setup_prepare()

        self._exchange_size_calc()

        return self._boundary_groups_contents_array, self._exchange_sizes

    @property
    def send_ranks(self):
        """
        Get the list of process ranks to send data to.

        :return: list of process ranks
        """
        if not self.check_valid():
            self._halo_setup_prepare()
        return self._send_list

    @property
    def recv_ranks(self):
        """
        Get list of process rands to expect to recv from.

        :return: list of process ranks.
        """
        if not self.check_valid():
            self._halo_setup_prepare()
        return self._recv_list



HALOS = None


# ===========================================================================

class CellSlice(object):
    def __getitem__(self, item):
        return item

# create a CellSlice object for easier halo definition.
Slice = CellSlice()


def create_halo_pairs_slice_halo(domain_in, slicexyz, direction):
    """
    Automatically create the pairs of cells for halos. Slices through 
    whole domain including halo cells.
    """

    cell_array = domain_in.cell_array
    extent = domain_in.extent

    xr = range(0, cell_array[0])[slicexyz[0]]
    yr = range(0, cell_array[1])[slicexyz[1]]
    zr = range(0, cell_array[2])[slicexyz[2]]

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
        if mpi.MPI_HANDLE.top[ix] == 0 and \
                        mpi.MPI_HANDLE.periods[ix] == 1 and \
                        direction[ix] == -1:

            shift[ix] = extent[ix]

        if mpi.MPI_HANDLE.top[ix] == mpi.MPI_HANDLE.dims[ix] - 1 and \
                        mpi.MPI_HANDLE.periods[ix] == 1 and \
                        direction[ix] == 1:

            shift[ix] = -1. * extent[ix]


    send_rank = mpi.MPI_HANDLE.shift((-1 * direction[0],
                                      -1 * direction[1],
                                      -1 * direction[2]))

    recv_rank = mpi.MPI_HANDLE.shift(direction)

    return b_cells, h_cells, shift, send_rank, recv_rank







class CartesianHaloSix(object):

    def __init__(self):
        self._timer = runtime.Timer(runtime.TIMER, 0, start=True)
        
        self._domain = cell.cell_list.domain
        self._ca_copy = [None, None, None]

        self._version = -1

        self._init = False

        # vars init
        self._boundary_cell_groups = host.Array(dtype=ctypes.c_int)
        self._boundary_groups_start_end_indices = host.Array(ncomp=7, dtype=ctypes.c_int)
        self._halo_cell_groups = host.Array(dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = host.Array(ncomp=7, dtype=ctypes.c_int)
        self._boundary_groups_contents_array = host.Array(dtype=ctypes.c_int)
        self._exchange_sizes = host.Array(ncomp=6, dtype=ctypes.c_int)

        self._send_ranks = host.Array(ncomp=6, dtype=ctypes.c_int)
        self._recv_ranks = host.Array(ncomp=6, dtype=ctypes.c_int)

        self._h_count = ctypes.c_int(0)
        self._t_count = ctypes.c_int(0)

        self._h_tmp = host.Array(ncomp=10, dtype=ctypes.c_int)
        self._b_tmp = host.Array(ncomp=10, dtype=ctypes.c_int)

        self.dir_counts = host.Array(ncomp=6, dtype=ctypes.c_int)


        self._halo_shifts = None

        # ensure first update
        self._boundary_cell_groups.inc_version(-1)
        self._boundary_groups_start_end_indices.inc_version(-1)
        self._halo_cell_groups.inc_version(-1)
        self._halo_groups_start_end_indices.inc_version(-1)
        self._boundary_groups_contents_array.inc_version(-1)
        self._exchange_sizes.inc_version(-1)

        self._setup()


        self._exchange_sizes_lib = None



    def _setup(self):
        """
        Internally setup the libraries for the calculation of exchange sizes.
        """
        pass

        self._init = True

    def _get_pairs(self):
        _cell_pairs = (

                # As these are the first exchange the halos cannot contain anything useful
                create_halo_pairs_slice_halo(self._domain, Slice[ 1, 1:-1 ,1:-1],(-1,0,0)),
                create_halo_pairs_slice_halo(self._domain, Slice[-2, 1:-1 ,1:-1 ],(1,0,0)),
                
                # As with the above no point exchanging anything extra in z direction
                create_halo_pairs_slice_halo(self._domain, Slice[::, 1, 1:-1],(0,-1,0)),
                create_halo_pairs_slice_halo(self._domain, Slice[::,-2, 1:-1],(0,1,0)),

                # Exchange all halo cells from x and y
                create_halo_pairs_slice_halo(self._domain, Slice[::,::,1],(0,0,-1)),
                create_halo_pairs_slice_halo(self._domain, Slice[::,::,-2],(0,0,1))
            )

        _bs = np.zeros(1, dtype=ctypes.c_int)
        _b = np.zeros(0, dtype=ctypes.c_int)

        _hs = np.zeros(1, dtype=ctypes.c_int)
        _h = np.zeros(0, dtype=ctypes.c_int)

        _s = np.zeros(0, dtype=ctypes.c_double)

        _len_h_tmp = 10
        _len_b_tmp = 10

        for hx, bhx in enumerate(_cell_pairs):

            print hx, bhx


            _len_b_tmp = max(_len_b_tmp, len(bhx[0]))
            _len_h_tmp = max(_len_h_tmp, len(bhx[1]))

            # Boundary and Halo start index.
            _bs = np.append(_bs, ctypes.c_int(len(bhx[0]) + _bs[-1] ))
            _hs = np.append(_hs, ctypes.c_int(len(bhx[1]) + _hs[-1] ))

            # Actual cell indices
            _b = np.append(_b, bhx[0])
            _h = np.append(_h, bhx[1])

            # Offset shifts for periodic boundary
            _s = np.append(_s, bhx[2])

            self._send_ranks[hx] = bhx[3]
            self._recv_ranks[hx] = bhx[4]

        if _len_b_tmp > self._b_tmp.ncomp:
            self._b_tmp.realloc(_len_b_tmp)

        if _len_h_tmp > self._h_tmp.ncomp:
            self._h_tmp.realloc(_len_h_tmp)


        # indices in array of cell indices
        self._boundary_groups_start_end_indices = host.Array(_bs, dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = host.Array(_hs, dtype=ctypes.c_int)

        # cell indices
        self._boundary_cell_groups = host.Array(_b, dtype=ctypes.c_int)
        self._halo_cell_groups = host.Array(_h, dtype=ctypes.c_int)

        # shifts for each direction.
        self._halo_shifts = host.Array(_s, dtype=ctypes.c_double)

        self._version = self._domain.cell_array.version


    def get_boundary_cell_groups(self):
        """
        Get the local boundary cells to pack for each halo. Formatted as an
        host.Array. Cells for halo 0 first followed by cells for halo 1 etc.
        Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting
        points within the first array.
        """

        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._boundary_cell_groups, self._boundary_groups_start_end_indices

    def get_halo_cell_groups(self):
        """
        Get the local halo cells to unpack into for each halo. Formatted as an
        cuda_base.Array. Cells for halo 0 first followed by cells for halo 1
        etc. Also returns an data.Array of 27 elements with the starting
        positions of each halo within the previous array.

        :return: Tuple, array of local halo cell indices to unpack into, array
        of starting points within the first array.
        """

        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._halo_cell_groups, self._halo_groups_start_end_indices

    def get_boundary_cell_contents_count(self):
        """
        Get the number of particles in the corresponding cells for each halo.
        These are needed such that the cell list can be created without
        inspecting the positions of recvd particles.

        :return: Tuple: Cell contents count for each cell in same order as
        local boundary cell list, Exchange sizes for each halo.
        """
        if not self._init:
            print "cuda_halo.CartesianHalo error. Library not initalised, " \
                  "this error means the internal setup failed."
            quit()

        self._exchange_sizes.zero()

        return self._boundary_groups_contents_array, self._exchange_sizes

    def get_position_shifts(self):
        """
        Calculate flag to determine if a boundary between processes is also
        a boundary in domain.
        """

        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._halo_shifts

    def get_send_ranks(self):
        """
        Get the mpi ranks to send to.
        """

        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._send_ranks

    def get_recv_ranks(self):
        """
        Get the mpi ranks to recv from.
        """

        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._recv_ranks

    def get_dir_counts(self):
        return self.dir_counts

    def exchange_cell_counts(self):
        """
        Exchange the contents count of cells between processes. This is
        provided as a method in halo to avoid repeated exchanging of cell
        occupancy counts if multiple ParticleDat objects are being
        communicated.
        """

        if self._exchange_sizes_lib is None:

            _es_args = '''
            const int f_MPI_COMM,             // F90 comm from mpi4py
            const int * RESTRICT SEND_RANKS,  // send directions
            const int * RESTRICT RECV_RANKS,  // recv directions
            const int * RESTRICT h_ind,       // halo indices
            const int * RESTRICT b_ind,       // local b indices
            const int * RESTRICT h_arr,       // h cell indices
            const int * RESTRICT b_arr,       // b cell indices
            int * RESTRICT ccc,               // cell contents count
            int * RESTRICT h_count,           // number of halo particles
            int * RESTRICT t_count,           // amount of tmp space needed
            int * RESTRICT h_tmp,             // tmp space for recving
            int * RESTRICT b_tmp,             // tmp space for sending
            int * RESTRICT dir_counts         // expected recv counts
            '''

            _es_header = '''
            #include <generic.h>
            #include <mpi.h>
            #include <iostream>
            using namespace std;
            #define RESTRICT %(RESTRICT)s

            extern "C" void HALO_ES_LIB(%(ARGS)s);
            '''

            _es_code = '''

            void HALO_ES_LIB(%(ARGS)s){
                *h_count = 0;
                *t_count = 0;

                // get mpi comm and rank
                MPI_Comm MPI_COMM = MPI_Comm_f2c(f_MPI_COMM);
                int rank = -1; MPI_Comm_rank( MPI_COMM, &rank );
                MPI_Status MPI_STATUS;

                // [W E] [N S] [O I]
                for( int dir=0 ; dir<6 ; dir++ ){

                    cout << "dir " << dir << "-------" << endl;

                    const int dir_s = b_ind[dir];             // start index
                    const int dir_c = b_ind[dir+1] - dir_s;   // cell count

                    const int dir_s_r = h_ind[dir];             // start index
                    const int dir_c_r = h_ind[dir+1] - dir_s_r; // cell count

                    int tmp_count = 0;
                    for( int ix=0 ; ix<dir_c ; ix++ ){
                        b_tmp[ix] = ccc[b_arr[dir_s + ix]];    // copy into
                                                               // send buffer


                        //cout << "\tcell: " << b_arr[dir_s + ix] << endl;
                        //cout << "\t\tcount: " << b_tmp[ix] << endl;

                        if (b_tmp[ix] == 946) { cout << "946 ccc: " << ccc[946] << endl;}

                        tmp_count += ccc[b_arr[dir_s + ix]];
                    }

                    cout << "\tcount 1: " << tmp_count << endl;


                    *t_count = MAX(*t_count, tmp_count);

                    // send b_tmp recv h_tmp

                    cout << "\tsendrecv " << dir_c << " " << dir_c_r << " " << SEND_RANKS[dir] << " " << RECV_RANKS[dir] << " " << rank << endl;


                    if(rank == RECV_RANKS[dir]){

                        for( int tx=0 ; tx < dir_c ; tx++ ){
                            h_tmp[tx] = b_tmp[tx];
                        }

                    } else {
                    MPI_Sendrecv ((void *) b_tmp, dir_c, MPI_INT,
                                  SEND_RANKS[dir], rank,
                                  (void *) h_tmp, dir_c_r, MPI_INT,
                                  RECV_RANKS[dir], RECV_RANKS[dir],
                                  MPI_COMM, &MPI_STATUS);
                    }
                    cout << "\tsendrecv completed" << endl;
                    // copy recieved values into correct places and sum;

                    tmp_count=0;
                    for( int ix=0 ; ix<dir_c_r ; ix++ ){
                        ccc[h_arr[dir_s_r + ix]] = h_tmp[ix];
                        *h_count += h_tmp[ix];
                        tmp_count += h_tmp[ix];
                    }
                    dir_counts[dir] = tmp_count;
                    *t_count = MAX(*t_count, tmp_count);


                    cout << "\tcount 2: " << tmp_count << endl;
                }

                return;
            }
            '''

            _es_dict = {'ARGS': _es_args,
                        'RESTRICT': build.MPI_CC.restrict_keyword}

            _es_header %= _es_dict
            _es_code %= _es_dict

            self._exchange_sizes_lib = build.simple_lib_creator(_es_header,
                                                                _es_code,
                                                                'HALO_ES_LIB',
                                                                CC=build.MPI_CC
                                                                )['HALO_ES_LIB']

        # End of creation code -----------------------------------------------

        # update internal arrays
        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        print str(mpi.MPI_HANDLE.rank) +  ' #' + ' before size exchange ' + 10*'#'

        self._exchange_sizes_lib(ctypes.c_int(mpi.MPI_HANDLE.fortran_comm),
                                 self._send_ranks.ctypes_data,
                                 self._recv_ranks.ctypes_data,
                                 self._halo_groups_start_end_indices.ctypes_data,
                                 self._boundary_groups_start_end_indices.ctypes_data,
                                 self._halo_cell_groups.ctypes_data,
                                 self._boundary_cell_groups.ctypes_data,
                                 cell.cell_list.cell_contents_count.ctypes_data,
                                 ctypes.byref(self._h_count),
                                 ctypes.byref(self._t_count),
                                 self._h_tmp.ctypes_data,
                                 self._b_tmp.ctypes_data,
                                 self.dir_counts.ctypes_data)

        print self.dir_counts.dat

        print str(mpi.MPI_HANDLE.rank) +  10*' #' + ' after size exchange  ' + 10*'#'

        return self._h_count.value, self._t_count.value




























