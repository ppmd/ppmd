import ctypes
import cell
import kernel
import build
import runtime
import mpi
import host

################################################################################################################
# HALO DEFINITIONS
################################################################################################################


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

        # CELL INDICES TO PACK (local boundary) =============================================================

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

        # ===========================================================================
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

        # =====================================================================================================================

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
        # ==========================================================================================================================

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

        # ==========================================================================================================================
        self._ca_copy = [cell.cell_list.domain.cell_array[0],
                         cell.cell_list.domain.cell_array[1],
                         cell.cell_list.domain.cell_array[2]]

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


    @property
    def get_boundary_cell_groups(self):
        """
        Get the local boundary cells to pack for each halo. Formatted as an data.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting points within the first array.
        """

        if not self.check_valid():
            self._halo_setup_prepare()
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
