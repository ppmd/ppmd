import ctypes
import numpy as np
import build
import data
import host
import kernel
import mpi
import pio
import runtime


class _AsFunc(object):
    """
    Instances of this class provide a callable to return the value of an attribute within an
    instance of another class.

    :arg instance: Instance of a class that has attributes.
    :arg str name: Name of attribute as string.
    """

    def __init__(self, instance, name):
        self._i = instance
        self._n = name

    def __call__(self):
        return getattr(self._i, self._n)



class BaseMDState(object):
    """
    Create an empty state to which particle properties such as position, velocities are added as
    attributes.

    """

    def __init__(self):
        # Registered particle dats.
        self.particle_dats = []

        # Local number of particles
        self._n = 0

        # do the ParticleDats have gaps in them?
        self.compressed = True
        """ Bool to determine if the held :class:`~data.ParticleDat` members have gaps in them. """

        self.uncompressed_n = False

        # move vars.

        self._move_dir_recv_totals = None
        self._move_dir_send_totals = None

        self._move_packing_shift_lib = None
        self._move_shift_array = host.NullDoubleArray

        self._move_send_buffer = None
        self._move_recv_buffer = None

        self._move_unpacking_lib = None
        self._move_empty_slots = []
        self._move_used_free_slot_count = None
        
        self._move_ncomp = None
        self._total_ncomp = None

        self.move_timer = runtime.Timer(runtime.TIMER, 0)
        self.move_timer2 = runtime.Timer(runtime.TIMER, 0)

        self._status = mpi.Status()

        # compressing vars
        self._compressing_lib = None
        self._compressing_n_new = None
        self._compressing_dyn_args = None

        self.compress_timer = runtime.Timer(runtime.TIMER, 0)


    def __setattr__(self, name, value):
        """
        Works the same as the default __setattr__ except that particle dats are registered upon being
        added. Added particle dats are registered in self.particle_dats.

        :param name: Name of parameter.
        :param value: Value of parameter.
        :return:
        """

        # Add to instance list of particle dats.
        if type(value) is data.ParticleDat:
            object.__setattr__(self, name, value)
            self.particle_dats.append(name)

            # Reset these to ensure that move libs are rebuilt.
            self._move_packing_lib = None
            self._move_send_buffer = None
            self._move_recv_buffer = None

            self._total_ncomp = 0
            self._move_ncomp = []
            for ixi, ix in enumerate(self.particle_dats):
                _dat = getattr(self, ix)
                self._move_ncomp.append(_dat.ncomp)
                self._total_ncomp += _dat.ncomp


        # Any other attribute.
        else:
            object.__setattr__(self, name, value)

    def as_func(self, name):
        """
        Returns a function handle to evaluate the required attribute.

        :arg str name: Name of attribute.
        :return: Callable that returns the value of attribute at the time of calling.)
        """
        return _AsFunc(self, name)

    @property
    def n(self):
        """
        :return: Local number of particles
        """
        return self._n

    @n.setter
    def n(self, value):
        """
        Set local number of particles.

        :arg value: New number of local particles.
        """
        self._n = int(value)
        for ix in self.particle_dats:
            _dat = getattr(self,ix)
            _dat.npart = int(value)
            _dat.halo_start_reset()
            #_dat.halo_start_set(int(value))

    def move_to_neighbour(self, ids=None, direction=None, shift=None):
        """
        Move particles to a neighbouring process.
        :return: bool as to whether move was successful.
        """

        self.move_timer.start()

        assert direction is not None, "move_to_neighbour error: No direction passed."
        # assert type(direction) is tuple, "move_to_neighbour error: passed direction should be a tuple."

        if type(direction) is not tuple:
            direction = mpi.recv_modifiers[direction]

        _send_rank = mpi.MPI_HANDLE.shift(direction, ignore_periods=True)
        _recv_rank = mpi.MPI_HANDLE.shift((-1 * direction[0],
                                          -1 * direction[1],
                                          -1 * direction[2]),
                                          ignore_periods=True)


        # Number of particles to send.
        if ids is None:
            _send_count = 0
        else:
            _send_count = len(ids)


        # Number of particles to expect to recv.
        _recv_count = np.array([-1], dtype=ctypes.c_int)

        _status = mpi.Status()

        # Exchange send sizes
        mpi.MPI_HANDLE.comm.Sendrecv(np.array([_send_count], dtype=ctypes.c_int),
                                     _send_rank,
                                     _send_rank,
                                     _recv_count,
                                     _recv_rank,
                                     mpi.MPI_HANDLE.rank,
                                     _status)

        _recv_count = int(_recv_count[0])


        # Create a send buffer.
        if self._move_send_buffer is None:
            self._move_send_buffer = host.Array(ncomp=_send_count * self._total_ncomp)


        elif _send_count * self._total_ncomp > self._move_send_buffer.ncomp:
            if runtime.VERBOSE.level > 2:
                print "rank", mpi.MPI_HANDLE.rank, ": move send buffer resized."

            self._move_send_buffer.realloc(_send_count * self._total_ncomp)


        if self._move_packing_lib is None or self._move_packing_shift_lib is None:

            _dynamic_dats = ''
            _dynamic_dats_shift = ''
            _space = ' ' * 16

            for ix, iy in zip(self.particle_dats, self._move_ncomp):

                # make case where ParticleDat has more than one component.
                if iy > 1:
                    _dynamic_dats += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                    _dynamic_dats += _space + 'SEND_BUFFER[index+ni] = %(NAME)s[(_ix*%(NCOMP)s)+ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                    _dynamic_dats += _space + '} \n'
                    _dynamic_dats += _space + 'index += %(NCOMP)s; \n' % {'NCOMP':iy}


                    if ix == 'positions':
                        _dynamic_dats_shift += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                        _dynamic_dats_shift += _space + 'SEND_BUFFER[index+ni] = %(NAME)s[(_ix*%(NCOMP)s)+ni] + SHIFT[ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                        _dynamic_dats_shift += _space + '} \n'
                        _dynamic_dats_shift += _space + 'index += %(NCOMP)s; \n' % {'NCOMP':iy}
                    else:
                        _dynamic_dats_shift += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                        _dynamic_dats_shift += _space + 'SEND_BUFFER[index+ni] = %(NAME)s[(_ix*%(NCOMP)s)+ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                        _dynamic_dats_shift += _space + '} \n'
                        _dynamic_dats_shift += _space + 'index += %(NCOMP)s; \n' % {'NCOMP':iy}

                else:
                    _dynamic_dats += _space + 'SEND_BUFFER[index] = %(NAME)s[_ix]; \n' % {'NAME':str(ix)}
                    _dynamic_dats += _space + 'index += 1; \n'

                    _dynamic_dats_shift += _space + 'SEND_BUFFER[index] = %(NAME)s[_ix]; \n' % {'NAME':str(ix)}
                    _dynamic_dats_shift += _space + 'index += 1; \n'


            _packing_code = '''
            int index = 0;
            for (int _idx = 0; _idx < end; _idx++){
                int _ix = _send_ids[_idx];
                \n%(DYNAMIC_DATS)s
            }
            ''' % {'DYNAMIC_DATS': _dynamic_dats}

            _packing_code_shift = '''
            int index = 0;
            for (int _idx = 0; _idx < end; _idx++){
                int _ix = _send_ids[_idx];
                \n%(DYNAMIC_DATS)s
            }
            ''' % {'DYNAMIC_DATS': _dynamic_dats_shift}


            _packing_headers = ['stdio.h']
            _packing_static_args = {'end':ctypes.c_int}
            self._packing_args = {'SEND_BUFFER':self._move_send_buffer, '_send_ids': host.NullIntArray}

            self._move_shift_array = host.Array(np.zeros(3), dtype=ctypes.c_double)

            self._packing_args_shift = {'SEND_BUFFER':self._move_send_buffer, '_send_ids': host.NullIntArray, 'SHIFT': self._move_shift_array}


            # Dynamic arguments dependant on how many particle dats there are.
            for idx, ix in enumerate(self.particle_dats):
                # existing dat in state
                self._packing_args['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)
                self._packing_args_shift['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)

            # create a unique but searchable name.
            _name = ''
            for ix in self.particle_dats:
                _name += '_' + str(ix)

            # make kernel
            _packing_kernel = kernel.Kernel('state_move_packing' + _name, _packing_code, None, _packing_headers, None, _packing_static_args)
            _packing_kernel_shift = kernel.Kernel('state_move_packing_shift' + _name, _packing_code_shift, None, _packing_headers, None, _packing_static_args)

            # make packing library
            self._move_packing_lib = build.SharedLib(_packing_kernel, self._packing_args)
            self._move_packing_shift_lib = build.SharedLib(_packing_kernel_shift, self._packing_args_shift)

        self.move_timer2.start()
        # Execute packing library.
        if shift is None:
            self._packing_args['_send_ids'] = host.Array(ids, dtype=ctypes.c_int)
            self._move_packing_lib.execute(static_args={'end': ctypes.c_int(len(ids))}, dat_dict=self._packing_args)
        else:
            self._packing_args_shift['_send_ids'] = host.Array(ids, dtype=ctypes.c_int)
            self._move_shift_array[0] = shift[0]
            self._move_shift_array[1] = shift[1]
            self._move_shift_array[2] = shift[2]


            self._move_packing_shift_lib.execute(static_args={'end': ctypes.c_int(len(ids))}, dat_dict=self._packing_args_shift)
        self.move_timer2.pause()

        # Create a recv buffer.
        if self._move_recv_buffer is None:
            self._move_recv_buffer = host.Array(ncomp=_send_count * self._total_ncomp)


        elif _recv_count * self._total_ncomp > self._move_recv_buffer.ncomp:
            if runtime.VERBOSE.level > 2:
                print "rank", mpi.MPI_HANDLE.rank, ": move recv buffer resized."

            self._move_recv_buffer.realloc(_recv_count * self._total_ncomp)


        # sending of particles.
        if _send_count > 0 and _recv_count > 0:
            mpi.MPI_HANDLE.comm.Sendrecv(self._move_send_buffer[0:_send_count * self._total_ncomp:],
                                         _send_rank,
                                         _send_rank,
                                         self._move_recv_buffer[0:_recv_count * self._total_ncomp:],
                                         _recv_rank,
                                         mpi.MPI_HANDLE.rank,
                                         _status)
        elif _send_count > 0:
                mpi.MPI_HANDLE.comm.Send(self._move_send_buffer[0:_send_count * self._total_ncomp:],
                                         _send_rank,
                                         _send_rank)
        elif _recv_count > 0:
                mpi.MPI_HANDLE.comm.Recv(self._move_recv_buffer[0:_recv_count * self._total_ncomp:],
                                         _recv_rank,
                                         mpi.MPI_HANDLE.rank,
                                         _status)
        '''
        print "=" * 50
        print self._move_send_buffer.dat
        print "-" * 50
        print self._move_recv_buffer.dat
        print "*" * 50
        '''


        # check that ParticleDats are large enough for the incoming particles.
        for ix in self.particle_dats:
            _dat = getattr(self, ix)
            if _dat.max_npart < _dat.npart + _recv_count - _send_count:
                _dat.resize(_dat.npart + _recv_count - _send_count)
                if runtime.VERBOSE.level > 2:
                    print "rank", mpi.MPI_HANDLE.rank, ix, ": particle dat resized."


        # free slots as array.
        for ix in range(_send_count):
            self._move_empty_slots.append(ids[ix])

        self._move_empty_slots.sort()

        _free_slots = host.Array(self._move_empty_slots, dtype=ctypes.c_int)
        _num_free_slots = _free_slots.ncomp


        if self._move_unpacking_lib is None:
            _dyn_dat_case = ''
            _space = ' ' * 16

            _cumulative_ncomp = 0

            for ixi, ix in enumerate(self.particle_dats):
                _dat = getattr(self, ix)
                if _dat.ncomp > 1:
                    _dyn_dat_case += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP': _dat.ncomp}
                    _dyn_dat_case += _space + '%(NAME)s[(pos*%(NCOMP)s)+ni] = _RECV_BUFFER[(ix*%(NCOMP_TOTAL)s)+%(NCOMP_START)s+ni]; \n' % {'NCOMP':_dat.ncomp, 'NAME':str(ix), 'NCOMP_TOTAL': self._total_ncomp, 'NCOMP_START': _cumulative_ncomp}
                    _dyn_dat_case += _space + '} \n'
                else:
                    _dyn_dat_case += _space + '%(NAME)s[pos] = _RECV_BUFFER[(ix*%(NCOMP_TOTAL)s)+%(NCOMP_START)s]; \n' % {'NAME':str(ix), 'NCOMP_TOTAL': self._total_ncomp, 'NCOMP_START': _cumulative_ncomp}

                _cumulative_ncomp += _dat.ncomp


            _unpacking_code = '''

            for(int ix = 0; ix < _recv_count; ix++){

                int pos;
                // prioritise filling spaces in dat.
                if (ix < _num_free_slots) {
                    pos = _free_slots[ix];
                } else {
                    pos = _prev_num_particles + ix - _num_free_slots;
                }

                \n%(DYN_DAT_CODE)s
            }
            ''' % {'DYN_DAT_CODE': _dyn_dat_case}

            _unpacking_static_args = {
                '_recv_count': ctypes.c_int,
                '_num_free_slots': ctypes.c_int,
                '_prev_num_particles': ctypes.c_int
            }

            _unpacking_dynamic_args = {
                '_free_slots': _free_slots,
                '_RECV_BUFFER': self._move_recv_buffer
            }

            for ix in self.particle_dats:
                # existing dat in state
                _unpacking_dynamic_args['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)

            _unpacking_headers = ['stdio.h']

            # create a unique but searchable name.
            _name = ''
            for ix in self.particle_dats:
                _name += '_' + str(ix)

            # make kernel
            _unpacking_kernel = kernel.Kernel('state_move_unpacking' + _name, _unpacking_code, None, _unpacking_headers, None, _unpacking_static_args)
            self._move_unpacking_lib = build.SharedLib(_unpacking_kernel, _unpacking_dynamic_args)




        _unpacking_dynamic_args = {'_free_slots': _free_slots, '_RECV_BUFFER': self._move_recv_buffer}

        for ix in self.particle_dats:
            # existing dat in state
            _unpacking_dynamic_args['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)

        self._move_unpacking_lib.execute(static_args={'_recv_count': ctypes.c_int(_recv_count),
                                                      '_num_free_slots': ctypes.c_int(_num_free_slots),
                                                      '_prev_num_particles': ctypes.c_int(self.n)},
                                         dat_dict=_unpacking_dynamic_args)


        #print "after move", self.n, _recv_count, _num_free_slots, _send_count

        if _recv_count < _num_free_slots:
            self.compressed = False
            self._move_empty_slots = self._move_empty_slots[_recv_count::]
        else:
            self.n = self.n + _recv_count - _num_free_slots
            self._move_empty_slots = []
        #print "after if", self.n, _recv_count, _num_free_slots, _send_count

        self.move_timer.pause()





    def move_to_neighbour_tmp(self, ids_directions_list=None, dir_send_totals=None, shifts=None):
        """
        Move particles using the linked list.

        :arg host.Array ids_directions_list(int): Linked list of ids from directions.
        :arg host.Array dir_send_totals(int): 26 Element array of number of particles traveling in each direction.
        :arg host.Array shifts(double): 73 element array of the shifts to apply when moving particles for the 26 directions.
        """

        if self._move_packing_lib is None:
            self._move_build_packing_lib()

        # Make/resize send buffer.
        if self._move_send_buffer is None:
            self._move_send_buffer = host.Array(ncomp=self._total_ncomp * dir_send_totals.dat.sum(), dtype=ctypes.c_double)
        elif self._move_send_buffer.ncomp < self._total_ncomp * dir_send_totals.dat.sum():
            self._move_send_buffer.realloc(self._total_ncomp * dir_send_totals.dat.sum())

        #Make recv sizes array.
        if self._move_dir_recv_totals is None:
            self._move_dir_recv_totals = host.Array(ncomp=26, dtype=ctypes.c_int)

        #exchange number of particles about to be sent.
        self._move_dir_send_totals = dir_send_totals
        self._move_exchange_send_recv_sizes()

        #resize recv buffer.
        _total = self._move_dir_recv_totals.dat.sum()
        if self._move_recv_buffer is None:
            self._move_recv_buffer = host.Array(ncomp=self._total_ncomp * _total, dtype=ctypes.c_double)
        elif self._move_recv_buffer.ncomp < self._total_ncomp * _total:
            self._move_recv_buffer.realloc(self._total_ncomp * _total)

        #pack particles to send.
        self._move_packing_shift_lib.execute(dat_dict={'SEND_BUFFER': self._move_send_buffer,
                                                       'SHIFT': shifts,
                                                       'direction_id_list': ids_directions_list})

        #exchange particle data.
        self._exchange_move_send_recv_buffers()





        return True

    def _exchange_move_send_recv_buffers(self):
        for ix in range(26):
            direction = mpi.recv_modifiers[ix]

            _send_rank = mpi.MPI_HANDLE.shift(direction, ignore_periods=True)
            _recv_rank = mpi.MPI_HANDLE.shift((-1 * direction[0],
                                               -1 * direction[1],
                                               -1 * direction[2]), ignore_periods=True)

            #TODO: empty slot list, starting points in send buffer for each direction.


            # sending of particles.
            if self._move_dir_send_totals[ix] > 0 and self._move_dir_recv_totals[ix] > 0:
                mpi.MPI_HANDLE.comm.Sendrecv(self._move_send_buffer[0:_send_count * self._total_ncomp:],
                                             _send_rank,
                                             _send_rank,
                                             self._move_recv_buffer[0:_recv_count * self._total_ncomp:],
                                             _recv_rank,
                                             mpi.MPI_HANDLE.rank,
                                             self._status)
            elif self._move_dir_send_totals[ix] > 0:
                    mpi.MPI_HANDLE.comm.Send(self._move_send_buffer[0:_send_count * self._total_ncomp:],
                                             _send_rank,
                                             _send_rank)
            elif self._move_dir_recv_totals[ix] > 0:
                    mpi.MPI_HANDLE.comm.Recv(self._move_recv_buffer[0:_recv_count * self._total_ncomp:],
                                             _recv_rank,
                                             mpi.MPI_HANDLE.rank,
                                             self._status)








    def _move_exchange_send_recv_sizes(self):
        """
        Exhange the sizes expected in the next particle move.
        """
        for ix in range(26):
            direction = mpi.recv_modifiers[ix]

            _send_rank = mpi.MPI_HANDLE.shift(direction, ignore_periods=True)
            _recv_rank = mpi.MPI_HANDLE.shift((-1 * direction[0],
                                               -1 * direction[1],
                                               -1 * direction[2]), ignore_periods=True)

            mpi.MPI_HANDLE.comm.Sendrecv(self._move_dir_send_totals[ix],
                                         _send_rank,
                                         _send_rank,
                                         self._move_dir_send_totals[ix],
                                         _recv_rank,
                                         mpi.MPI_HANDLE.rank,
                                         self._status)

    def _move_build_packing_lib(self):
        """
        Build the library to packing particles to send.
        """

        _dynamic_dats_shift = ''
        _space = ' ' * 16

        for ix, iy in zip(self.particle_dats, self._move_ncomp):

            # make case where ParticleDat has more than one component.
            if iy > 1:

                if ix == 'positions':
                    _dynamic_dats_shift += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                    _dynamic_dats_shift += _space + 'SEND_BUFFER[index+ni] = %(NAME)s[(_ix*%(NCOMP)s)+ni] + SHIFT[(_dir*3)+ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                    _dynamic_dats_shift += _space + '} \n'
                    _dynamic_dats_shift += _space + 'index += %(NCOMP)s; \n' % {'NCOMP':iy}
                else:
                    _dynamic_dats_shift += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                    _dynamic_dats_shift += _space + 'SEND_BUFFER[index+ni] = %(NAME)s[(_ix*%(NCOMP)s)+ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                    _dynamic_dats_shift += _space + '} \n'
                    _dynamic_dats_shift += _space + 'index += %(NCOMP)s; \n' % {'NCOMP':iy}

            else:
                _dynamic_dats_shift += _space + 'SEND_BUFFER[index] = %(NAME)s[_ix]; \n' % {'NAME':str(ix)}
                _dynamic_dats_shift += _space + 'index += 1; \n'


        _packing_code_shift = '''
        // Next free space in send buffer.
        int index = 0;

        //loop over the send directions.
        for(int _dir = 0; _dir < 26; _dir++){

            //traverse linked list.
            int _ix = direction_id_list[_dir];

            while(_ix > -1){

                //Generate code based on ParticleDats

                \n%(DYNAMIC_DATS)s

                _ix = direction_id_list[_ix + 1];
            }
        }
        ''' % {'DYNAMIC_DATS': _dynamic_dats_shift}

        self._packing_args_shift = {'SEND_BUFFER':host.NullDoubleArray,
                                    'SHIFT': host.NullDoubleArray,
                                    'direction_id_list': host.NullIntArray}


        # Dynamic arguments dependant on how many particle dats there are.
        for idx, ix in enumerate(self.particle_dats):
            # existing dat in state
            self._packing_args_shift['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)

        # create a unique but searchable name.
        _name = ''
        for ix in self.particle_dats:
            _name += '_' + str(ix)

        # make kernel
        _packing_kernel_shift = kernel.Kernel('state_move_packing_shift' + _name, _packing_code_shift, None, ['stdio.h'], None, None)

        # make packing library
        self._move_packing_shift_lib = build.SharedLib(_packing_kernel_shift, self._packing_args_shift)























    def compress_particle_dats(self):
        """
        Compress the particle dats held in the state. Compressing removes empty rows.
        """

        #print self.compressed


        self._compressing_n_new = host.Array([0], dtype=ctypes.c_int)
        self._compressing_slots = host.Array(self._move_empty_slots, dtype=ctypes.c_int)

        if self._compressing_lib is None:

            _dyn_dat_case = ''
            _space = ' ' * 16

            for ixi, ix in enumerate(self.particle_dats):
                _dat = getattr(self, ix)
                if _dat.ncomp > 1:
                    _dyn_dat_case += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP': _dat.ncomp}
                    _dyn_dat_case += _space + '%(NAME)s[(slot_to_fill*%(NCOMP)s)+ni] = %(NAME)s[found_index*%(NCOMP)s+ni]; \n' % {'NCOMP':_dat.ncomp, 'NAME':str(ix)}
                    _dyn_dat_case += _space + '} \n'
                else:
                    _dyn_dat_case += _space + '%(NAME)s[slot_to_fill] = %(NAME)s[found_index]; \n' % {'NAME':str(ix)}


            _static_args = {
                'slots_to_fill_in': ctypes.c_int,
                'n_new_in': ctypes.c_int
            }

            self._compressing_dyn_args = {
                'slots': self._compressing_slots,
                'n_new_out': self._compressing_n_new
            }

            _compressing_code = '''
            //printf("-------- compress start ------------ \\n");


            int slots_to_fill = slots_to_fill_in;
            int n_new = n_new_in;

            int last_slot;
            int last_slot_lookup_index = slots_to_fill - 1;

            int slot_to_fill_index = 0;

            int slot_to_fill = -1;

            //printf("n_new=%%d, slots_to_fill=%%d, last_slot_lookup_index=%%d \\n", n_new, slots_to_fill, last_slot_lookup_index);


            // Whilst there are slots to fill and the current slot is not past the end of the array.
            if (n_new > 0) {
                while ( (slot_to_fill_index <= last_slot_lookup_index) && (slots[slot_to_fill_index] < n_new) ){

                    // get first empty slot in particle dats.
                    slot_to_fill = slots[slot_to_fill_index];

                    int found_index = -1;

                    //loop from end to empty slot
                    for (int iy = n_new - 1; iy > slot_to_fill; iy--){

                        //printf("iy=%%d, slots[last_slot_lookup_index]=%%d \\n", iy, slots[last_slot_lookup_index]);

                        if (iy == slots[last_slot_lookup_index]){
                            n_new = iy;
                            last_slot_lookup_index--;
                            printf("n_new=%%d \\n", n_new);
                        } else {
                            found_index = iy;
                            break;
                        }

                    }

                    if (found_index > 0){

                        \n%(DYN_DAT_CODE)s

                        n_new = found_index;


                    } else {
                        //printf("last slot = %%d \\n", slots[last_slot_lookup_index]);

                        n_new = slots[last_slot_lookup_index];
                        break;
                    }


                    slot_to_fill_index++;
                }
            }
            //printf("n_new_out=%%d \\n", n_new);
            n_new_out[0] = n_new;

            //printf("-------- compress end ------------ \\n");
            ''' % {'DYN_DAT_CODE': _dyn_dat_case}

            # Add ParticleDats to pointer arguments.
            for idx, ix in enumerate(self.particle_dats):
                self._compressing_dyn_args['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)

            _compressing_kernel = kernel.Kernel('ParticleDat_compressing_lib', _compressing_code, headers=['stdio.h'], static_args=_static_args)


            self._compressing_lib = build.SharedLib(_compressing_kernel, self._compressing_dyn_args)

        if self.compressed is True:
            return
        else:

            self._compressing_dyn_args['slots'] = self._compressing_slots
            self._compressing_dyn_args['n_new_out'] = self._compressing_n_new

            #print self.n, self._compressing_n_new[0], "compressing slots=", self._compressing_slots.dat
            self.compress_timer.start()
            self._compressing_lib.execute(static_args={'slots_to_fill_in': ctypes.c_int(self._compressing_slots.ncomp), 'n_new_in': ctypes.c_int(self.n)},
                                          dat_dict=self._compressing_dyn_args)


            #print self.n, self._compressing_n_new[0]

            self.n = self._compressing_n_new[0]
            self.compressed = True
            self._move_empty_slots = []
            self.compress_timer.pause()









