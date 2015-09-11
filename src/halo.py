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


class HaloCartesianSingleProcess(object):
    """
    Class to contain and control cartesian halo transfers.
    
    :arg int nt: Total number of particles in system.
    :arg MPI_Comm mpi_handle: MPI communicator.
    :arg int rank: Process rank
    :arg tuple top: location of process in cartesian topology
    :arg tuple dims: number of processes in each coordinate direction.
    :arg list cell_array: List of local cell array.
    :arg host.Array extent: Global extent of system.
    
    """

    def __init__(self, nt=1):

        self._NT = nt


        self.timer = runtime.Timer(runtime.TIMER, 0, start=True)

        self._MPI_handle = mpi.MPI_HANDLE

        self._MPI = self._MPI_handle.comm
        self._rank = self._MPI_handle.rank
        self._top = self._MPI_handle.top
        self._dims = self._MPI_handle.dims

        self._MPIstatus = mpi.Status()

        self._ca = cell.cell_list.domain.cell_array
        self._ca_copy = [None, None, None]

        # n.B. this is a global extent.
        self._extent = cell.cell_list.domain.extent

        self._halos = []

        # Will need at least 3 components for positions
        self._nc = 3
        self._cell_contents_count = cell.cell_list.cell_contents_count
        self._cell_list = cell.cell_list.cell_list

        self._halo_setup_prepare()

        self._create_packing_pointer_array()
        self._create_packing_lib()

        self.timer.stop("halo setup time.")

    def exchange(self, data_in):
        """
        Exchange data using halos.
        
        :arg host.Array self._cell_contents_count: Number of particles in each local cell.
        :arg host.Array self._cell_list: Local particle cell list.
        :arg host.Matrix type data_in: Particle data to be halo exchanged.
        
        """


        if self._nc < data_in.ncomp:
            self._nc = data_in.ncomp
            self._create_packing_pointer_array()

        self._nc = data_in.ncomp

        self.timer.start()

        '''Get new storage sizes'''
        self._exchange_size_calc()

        '''Reset halo starting points in input data'''
        data_in.halo_start_reset()

        '''Position dependent switch, make more concrete with flag in particle dat'''
        if data_in.name == 'positions':
            self._cell_shifts_array = self._cell_shifts_array_pbc
        else:
            self._cell_shifts_array = self._cell_shifts_array_zero

        '''Pack data for each direction'''
        _static_args = {
            'PPA': self._packing_pointers.ctypes_data,
            'cell_start': ctypes.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end]),
            'ncomp': ctypes.c_int(data_in.ncomp)
        }
        _args = {
            'q': cell.cell_list.cell_list,
            'CCA_I': self._cell_contents_array_index,
            'CIA': self._cell_indices_array,
            'CSA': self._cell_shifts_array,
            'data_buffer': data_in
        }
        self._packing_lib.execute(static_args=_static_args, dat_dict=_args)

        '''Exchange in each direction.'''

        self._cell_contents_recv.zero()

        # print self._rank, self._send_list

        for i in range(26):

            # SIZES ------------------------------------------------------------------------------------------

            if self._send_list[i] > -1 and self._recv_list[i] > -1:

                self._MPI.Sendrecv(self._cell_contents_array[
                                   self._cell_contents_array_index[i]:self._cell_contents_array_index[i + 1]:],
                                   self._send_list[i],
                                   self._send_list[i],
                                   self._cell_contents_recv[
                                   self._cell_contents_recv_array_index[i]:self._cell_contents_recv_array_index[
                                       i + 1]:],
                                   self._recv_list[i],
                                   self._rank,
                                   self._MPIstatus)

            elif self._send_list[i] > -1:

                self._MPI.Send(self._cell_contents_array[
                               self._cell_contents_array_index[i]:self._cell_contents_array_index[i + 1]:],
                               self._send_list[i],
                               self._send_list[i])

            elif self._recv_list[i] > -1:

                self._MPI.Recv(self._cell_contents_recv[
                               self._cell_contents_recv_array_index[i]:self._cell_contents_recv_array_index[i + 1]:],
                               self._recv_list[i],
                               self._rank,
                               self._MPIstatus)
            # DATA ------------------------------------------------------------------------------------------

            if self._send_list[i] > -1 and self._recv_list[i] > -1:

                self._MPI.Sendrecv(self._send_buffers[i].dat[0:self._exchange_sizes[i]:1, ::],
                                   self._send_list[i],
                                   self._send_list[i],
                                   data_in.dat[data_in.halo_start::, ::],
                                   self._recv_list[i],
                                   self._rank,
                                   self._MPIstatus)

                _shift = self._MPIstatus.Get_count(mpi.mpi_map[data_in.dtype])
                data_in.halo_start_shift(_shift / self._nc)

            elif self._send_list[i] > -1:

                self._MPI.Send(self._send_buffers[i].dat[0:self._exchange_sizes[i]:1, ::],
                               self._send_list[i],
                               self._send_list[i])

            elif self._recv_list[i] > -1:

                self._MPI.Recv(data_in.dat[data_in.halo_start::, ::],
                               self._recv_list[i],
                               self._rank,
                               self._MPIstatus)

                _shift = self._MPIstatus.Get_count(mpi.mpi_map[data_in.dtype])

                data_in.halo_start_shift(_shift / self._nc)

        '''sort local cells after exchange'''
        if data_in.name == 'positions':
            self._cell_sort_loop.execute(
                {'q': cell.cell_list.cell_list, 'LCI': self._local_cell_indices_array, 'CRC': self._cell_contents_recv, 'CCC': cell.cell_list.cell_contents_count},
                {'CC': ctypes.c_int(self._cell_contents_recv.ncomp), 'shift': ctypes.c_int(data_in.npart),
                 'end': ctypes.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end])})

        self.timer.pause()

    def _create_packing_pointer_array(self):
        self._packing_pointers = host.PointerArray(length=26, dtype=ctypes.c_double)

        self._send_buffers = []

        for ix in range(26):
            # create send buffer when needed
            if self._send_list[ix] > -1:
                self._send_buffers.append(host.Matrix(self._NT, self._nc, dtype=ctypes.c_double))
                self._packing_pointers[ix] = self._send_buffers[ix].ctypes_data
            else:
                self._send_buffers.append(None)

    def _create_packing_lib(self):

        _packing_code = '''
        //Loop over directions
        for(int ix = 0; ix<26; ix++ ){
            
            //get the start and end indices in the array containing cell indices
            const int start = CCA_I[ix];
            const int end = CCA_I[ix+1];
            int index = 0;
            
            //loop over cells
            for(int iy = start; iy < end; iy++){
                
                // current cell
                int c_i = CIA[iy];
                
                // first particle
                int iz = q[cell_start+c_i];
                
                while(iz > -1){
                    
                    // loop over the number of components for particle dat.
                    for(int cx = 0; cx<ncomp;cx++){
                        PPA[ix][LINIDX_2D(ncomp,index,cx)] = data_buffer[LINIDX_2D(ncomp,iz,cx)] + CSA[LINIDX_2D(ncomp,ix,cx)];
                    }
                    index++;
                    iz = q[iz];
                }
            }
        }
        
        '''

        _static_args = {
            'PPA': 'doublepointerpointer',  # self._packing_pointers.ctypes_data
            'cell_start': ctypes.c_int,  # ctypes.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end])
            'ncomp': ctypes.c_int  # ctypes.c_int(data.ncomp)
        }

        _args = {
            'q': host.NullIntArray,  # cell.cell_list.cell_list.ctypes_data
            'CCA_I': self._cell_contents_array_index,
            'CIA': self._cell_indices_array,
            'CSA': host.NullDoubleArray,  # self._cell_shifts_array
            'data_buffer': host.NullDoubleArray,  # data
        }

        _headers = ['stdio.h']
        _kernel = kernel.Kernel('HaloPackingCode', _packing_code, None, _headers, None, _static_args)
        self._packing_lib = build.SharedLib(_kernel, _args)

    def _exchange_size_calc(self):
        """
        for i,x in enumerate(self._cell_indices):
            self._exchange_sizes[i]=sum([y[1] for y in enumerate(self._cell_contents_count) if y[0] in x])
        """


        _args = {
            'CCC': cell.cell_list.cell_contents_count,
            'ES': self._exchange_sizes,
            'CI': self._cell_indices_array,
            'CIL': self._cell_indices_len,
            'CCA': self._cell_contents_array,
        }

        self._exchange_sizes_lib.execute(dat_dict=_args)

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
            _tr = self._MPI_handle.shift([-1 * ix[0], -1 * ix[1], -1 * ix[2]])
            self._send_list.append(_tr)

            _tr = self._MPI_handle.shift(ix)
            self._recv_list.append(_tr)

        '''Array to store the number of particles to exchange for each halo'''
        self._exchange_sizes = host.Array(ncomp=26, dtype=ctypes.c_int)


        # CELL INDICES TO PACK ===========================================================================================

        _E = self._ca[0] * self._ca[1] * (self._ca[2] - 1) - self._ca[0] - 1
        _TS = _E - self._ca[0] * (self._ca[1] - 2) + 2

        _BE = self._ca[0] * (2 * self._ca[1] - 1) - 1
        _BS = self._ca[0] * (self._ca[1] + 1) + 1

        _tmp4 = []
        for ix in range(self._ca[1] - 2):
            _tmp4 += range(_TS + ix * self._ca[0], _TS + (ix + 1) * self._ca[0] - 2, 1)

        _tmp10 = []
        for ix in range(self._ca[2] - 2):
            _tmp10 += range(_BE - self._ca[0] + 2 + ix * self._ca[0] * self._ca[1],
                            _BE + ix * self._ca[0] * self._ca[1], 1)

        _tmp12 = []
        for ix in range(self._ca[2] - 2):
            _tmp12 += range(_BS + self._ca[0] - 3 + ix * self._ca[0] * self._ca[1],
                            _BE + ix * self._ca[0] * self._ca[1], self._ca[0])

        _tmp13 = []
        for ix in range(self._ca[2] - 2):
            _tmp13 += range(_BS + ix * self._ca[0] * self._ca[1], _BE + ix * self._ca[0] * self._ca[1], self._ca[0])

        _tmp15 = []
        for ix in range(self._ca[2] - 2):
            _tmp15 += range(_BS + ix * self._ca[0] * self._ca[1],
                            _BS + self._ca[0] - 2 + ix * self._ca[0] * self._ca[1], 1)

        _tmp21 = []
        for ix in range(self._ca[1] - 2):
            _tmp21 += range(_BS + ix * self._ca[0], _BS + (ix + 1) * self._ca[0] - 2, 1)

        self._cell_indices = [
            [_E - 1],
            range(_E - self._ca[0] + 2, _E, 1),
            [_E - self._ca[0] + 2],
            range(_TS + self._ca[0] - 3, _E, self._ca[0]),
            _tmp4,
            range(_TS, _E, self._ca[0]),
            [_TS + self._ca[0] - 3],
            range(_TS, _TS + self._ca[0] - 2, 1),
            [_TS],

            range(_BE - 1, _E, self._ca[0] * self._ca[1]),
            _tmp10,
            range(_BE - self._ca[0] + 2, _E, self._ca[0] * self._ca[1]),
            _tmp12,
            _tmp13,
            range(_BS + self._ca[0] - 3, _E, self._ca[0] * self._ca[1]),
            _tmp15,
            range(_BS, _E, self._ca[0] * self._ca[1]),

            [_BE - 1],
            range(_BE - self._ca[0] + 2, _BE, 1),
            [_BE - self._ca[0] + 2],
            range(_BS + self._ca[0] - 3, _BE, self._ca[0]),
            _tmp21,
            range(_BS, _BE, self._ca[0]),
            [_BS + self._ca[0] - 3],
            range(_BS, _BS + self._ca[0] - 2, 1),
            [_BS]
        ]

        # =====================================================================================================================
        # LOCAL CELL INDICES TO SORT INTO

        _LE = self._ca[0] * self._ca[1] * self._ca[2]
        _LTS = _LE - self._ca[0] * self._ca[1]

        _LBE = self._ca[0] * self._ca[1]
        _LBS = 0

        _Ltmp4 = []
        for ix in range(self._ca[1] - 2):
            _Ltmp4 += range(_LBS + 1 + (ix + 1) * self._ca[0], _LBS + (ix + 2) * self._ca[0] - 1, 1)

        _Ltmp10 = []
        for ix in range(self._ca[2] - 2):
            _Ltmp10 += range(_LBS + 1 + (ix + 1) * self._ca[0] * self._ca[1],
                             _LBS + (ix + 1) * self._ca[0] * self._ca[1] + self._ca[0] - 1, 1)

        _Ltmp12 = []
        for ix in range(self._ca[2] - 2):
            _Ltmp12 += range(_LBS + self._ca[0] + (ix + 1) * self._ca[0] * self._ca[1],
                             _LBS + self._ca[0] * (self._ca[1] - 1) + (ix + 1) * self._ca[0] * self._ca[1], self._ca[0])

        _Ltmp13 = []
        for ix in range(self._ca[2] - 2):
            _Ltmp13 += range(_LBS + 2 * self._ca[0] - 1 + (ix + 1) * self._ca[0] * self._ca[1],
                             _LBE + (ix + 1) * self._ca[0] * self._ca[1] - 3, self._ca[0])

        _Ltmp15 = []
        for ix in range(self._ca[2] - 2):
            _Ltmp15 += range(_LBE + (ix + 1) * self._ca[0] * self._ca[1] - self._ca[0] + 1,
                             _LBE + (ix + 1) * self._ca[0] * self._ca[1] - 1, 1)

        _Ltmp21 = []
        for ix in range(self._ca[1] - 2):
            _Ltmp21 += range(_LTS + (ix + 1) * self._ca[0] + 1, _LTS + (ix + 2) * self._ca[0] - 1, 1)

        self._local_cell_indices = [
            [_LBS],
            range(_LBS + 1, _LBS + self._ca[0] - 1, 1),
            [_LBS + self._ca[0] - 1],
            range(_LBS + self._ca[0], _LBS + self._ca[0] * (self._ca[1] - 2) + 1, self._ca[0]),
            _Ltmp4,
            range(_LBS + 2 * self._ca[0] - 1, _LBE - self._ca[0], self._ca[0]),
            [_LBE - self._ca[0]],
            range(_LBE - self._ca[0] + 1, _LBE - 1, 1),
            [_LBE - 1],

            range(_LBS + self._ca[0] * self._ca[1], _LBS + (self._ca[2] - 2) * self._ca[0] * self._ca[1] + 1,
                  self._ca[0] * self._ca[1]),
            _Ltmp10,
            range(_LBE + self._ca[0] - 1, _LBE + (self._ca[2] - 2) * self._ca[0] * self._ca[1] + self._ca[0] - 1,
                  self._ca[0] * self._ca[1]),
            _Ltmp12,
            _Ltmp13,
            range(_LBE + self._ca[0] * (self._ca[1] - 1), _LE - self._ca[0] * self._ca[1], self._ca[0] * self._ca[1]),
            _Ltmp15,
            range(_LBE + self._ca[0] * self._ca[1] - 1, _LE - self._ca[0] * self._ca[1], self._ca[0] * self._ca[1]),

            [_LTS],
            range(_LTS + 1, _LTS + self._ca[0] - 1, 1),
            [_LTS + self._ca[0] - 1],
            range(_LTS + self._ca[0], _LTS + self._ca[0] * (self._ca[1] - 2) + 1, self._ca[0]),
            _Ltmp21,
            range(_LTS + 2 * self._ca[0] - 1, _LE - self._ca[0], self._ca[0]),
            [_LE - self._ca[0]],
            range(_LE - self._ca[0] + 1, _LE - 1, 1),
            [_LE - 1],
        ]

        # =====================================================================================================================


        '''How many cells are in each halo (internal to pack)'''
        self._cell_indices_len = host.Array(ncomp=26, dtype=ctypes.c_int)


        '''How many cells are in each halo (external to recv)'''
        self._cell_indices_recv_len = host.Array(range(26), dtype=ctypes.c_int)

        _tmp_list = []
        _tmp_list_local = []
        for ix in range(26):
            # determine internal cells to pack
            if self._send_list[ix] > -1:
                self._cell_indices_len[ix] = len(self._cell_indices[ix])
                _tmp_list += self._cell_indices[ix]
            else:
                self._cell_indices_len[ix] = 0

            # detmine halo cells to unpack into
            if self._recv_list[ix] > -1:
                self._cell_indices_recv_len[ix] = len(self._local_cell_indices[ix])
                _tmp_list_local += self._local_cell_indices[ix]
            else:
                self._cell_indices_recv_len[ix] = 0

        '''Array containing the internal cell indices'''
        self._cell_indices_array = host.Array(_tmp_list, dtype=ctypes.c_int)

        '''Array containing the halo cell indices'''
        self._local_cell_indices_array = host.Array(_tmp_list_local, dtype=ctypes.c_int)

        '''create cell contents array for each halo that is being sent.'''
        self._cell_contents_array = host.Array(ncomp=self._cell_indices_array.ncomp, dtype=ctypes.c_int)

        '''create cell contents array for each halo that are being recv'd.'''
        self._cell_contents_recv = host.Array(ncomp=self._local_cell_indices_array.ncomp, dtype=ctypes.c_int)

        '''SEND: create list to extract start and end points for each halo from above lists.'''
        self._cell_contents_array_index = host.Array(ncomp=27, dtype=ctypes.c_int)

        '''RECV: create list to extract start and end points for each halo from above lists.'''
        self._cell_contents_recv_array_index = host.Array(ncomp=27, dtype=ctypes.c_int)




        '''Indices for sending'''
        _start_index = 0
        self._cell_contents_array_index[0] = 0
        for ix in range(26):
            _start_index += self._cell_indices_len[ix]
            self._cell_contents_array_index[ix + 1] = _start_index

        '''Indices for receving'''
        _start_index = 0
        self._cell_contents_recv_array_index[0] = 0
        for ix in range(26):
            _start_index += self._cell_indices_recv_len[ix]
            self._cell_contents_recv_array_index[ix + 1] = _start_index




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

        '''Code to sort incoming halo particles into cell list '''
        # ==========================================================================================================================

        _cell_sort_code = '''

        int index = shift;
        for(int ix = 0; ix < CC; ix++){
            
            //get number of particles
            const int _tmp = CRC[ix];
            
            if (_tmp>0){
                
                //first index in cell region of cell list.
                q[end+LCI[ix]] = index;
                CCC[LCI[ix]] = _tmp;

                //start at first particle in halo cell, work forwards
                for(int iy = 0; iy < _tmp-1; iy++){
                    q[index+iy]=index+iy+1;
                }
                q[index+_tmp-1] = -1;
            }
            
            
            index += CRC[ix];

        }
        '''

        _static_args = {'CC': ctypes.c_int, 'shift': ctypes.c_int, 'end': ctypes.c_int}

        _cell_sort_dict = {
            'q': host.NullIntArray,
            'LCI': self._local_cell_indices_array,
            'CRC': self._cell_contents_recv,
            'CCC': self._cell_contents_count
        }

        _cell_sort_kernel = kernel.Kernel('halo_cell_list_method', _cell_sort_code, headers=['stdio.h'],
                                          static_args=_static_args)
        self._cell_sort_loop = build.SharedLib(_cell_sort_kernel, _cell_sort_dict)
        # ==========================================================================================================================

        '''
        Xl 0, Xu 1
        Yl 2, Yu 3
        Zl 4, Zu 5
        '''

        '''Calculate flag to determine if a boundary between processes is also a boundary in domain.'''
        _bc_flag = [0, 0, 0, 0, 0, 0]
        for ix in range(3):
            if self._top[ix] == 0:
                _bc_flag[2 * ix] = 1
            if self._top[ix] == self._dims[ix] - 1:
                _bc_flag[2 * ix + 1] = 1

        '''Shifts to apply to positions when exchanging over boundaries.'''
        _cell_shifts = [
            [-1 * self._extent[0] * _bc_flag[1], -1 * self._extent[1] * _bc_flag[3],
             -1 * self._extent[2] * _bc_flag[5]],
            [0., -1 * self._extent[1] * _bc_flag[3], -1 * self._extent[2] * _bc_flag[5]],
            [self._extent[0] * _bc_flag[0], -1 * self._extent[1] * _bc_flag[3], -1 * self._extent[2] * _bc_flag[5]],
            [-1 * self._extent[0] * _bc_flag[1], 0., -1 * self._extent[2] * _bc_flag[5]],
            [0., 0., -1 * self._extent[2] * _bc_flag[5]],
            [self._extent[0] * _bc_flag[0], 0., -1 * self._extent[2] * _bc_flag[5]],
            [-1 * self._extent[0] * _bc_flag[1], self._extent[1] * _bc_flag[2], -1 * self._extent[2] * _bc_flag[5]],
            [0., self._extent[1] * _bc_flag[2], -1 * self._extent[2] * _bc_flag[5]],
            [self._extent[0] * _bc_flag[0], self._extent[1] * _bc_flag[2], -1 * self._extent[2] * _bc_flag[5]],

            [-1 * self._extent[0] * _bc_flag[1], -1 * self._extent[1] * _bc_flag[3], 0.],
            [0., -1 * self._extent[1] * _bc_flag[3], 0.],
            [self._extent[0] * _bc_flag[0], -1 * self._extent[1] * _bc_flag[3], 0.],
            [-1 * self._extent[0] * _bc_flag[1], 0., 0.],
            [self._extent[0] * _bc_flag[0], 0., 0.],
            [-1 * self._extent[0] * _bc_flag[1], self._extent[1] * _bc_flag[2], 0.],
            [0., self._extent[1] * _bc_flag[2], 0.],
            [self._extent[0] * _bc_flag[0], self._extent[1] * _bc_flag[2], 0.],

            [-1 * self._extent[0] * _bc_flag[1], -1 * self._extent[1] * _bc_flag[3], self._extent[2] * _bc_flag[4]],
            [0., -1 * self._extent[1] * _bc_flag[3], self._extent[2] * _bc_flag[4]],
            [self._extent[0] * _bc_flag[0], -1 * self._extent[1] * _bc_flag[3], self._extent[2] * _bc_flag[4]],
            [-1 * self._extent[0] * _bc_flag[1], 0., self._extent[2] * _bc_flag[4]],
            [0., 0., self._extent[2] * _bc_flag[4]],
            [self._extent[0] * _bc_flag[0], 0., self._extent[2] * _bc_flag[4]],
            [-1 * self._extent[0] * _bc_flag[1], self._extent[1] * _bc_flag[2], self._extent[2] * _bc_flag[4]],
            [0., self._extent[1] * _bc_flag[2], self._extent[2] * _bc_flag[4]],
            [self._extent[0] * _bc_flag[0], self._extent[1] * _bc_flag[2], self._extent[2] * _bc_flag[4]]
        ]

        '''make scalar array object from above shifts'''
        _tmp_list_local = []
        '''zero scalar array for data that is not position dependent'''
        _tmp_zero = range(26)
        for ix in range(26):
            _tmp_list_local += _cell_shifts[ix]
            _tmp_zero[ix] = 0

        self._cell_shifts_array_pbc = host.Array(_tmp_list_local, dtype=ctypes.c_double)
        self._cell_shifts_array_zero = host.Array(_tmp_zero, dtype=ctypes.c_double)

        self._ca_copy = [cell.cell_list.domain.cell_array[0],
                         cell.cell_list.domain.cell_array[1],
                         cell.cell_list.domain.cell_array[2]]

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
    def local_boundary_cells_for_halos(self):
        """
        Get the local boundary cells to pack for each halo. Formatted as an data.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting points within the first
        array.
        """

        if not self.check_valid():
            self._halo_setup_prepare()
        return self._cell_indices_array, self._cell_contents_array_index


    @property
    def local_halo_cells(self):
        """
        Get the local halo cells to unpack into for each halo. Formatted as an data.Array. Cells for halo
        0 first followed by cells for halo 1 etc. Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local halo cell indices to unpack into, array of starting points within the first
        array.
        """

        if not self.check_valid():
            self._halo_setup_prepare()
        return self._local_cell_indices_array, self._cell_contents_recv_array_index

    @property
    def local_boundary_cell_contents_count(self):
        """
        Get the number of particles in the corresponding cells for each halo. These are needed such that
        the cell list can be created without inspecting the positions of recvd particles.
        :return: Tuple: Cell contents count for each cell in same order as local boundary cell list.,
        Exchange sizes for each halo.
        """

        self._exchange_size_calc()

        return self._cell_contents_array, self._exchange_sizes

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
