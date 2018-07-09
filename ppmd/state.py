from __future__ import print_function, division, absolute_import

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np
# package level
from ppmd import data, cell, host, mpi, runtime, halo
import ppmd.lib.build as build

SUM = mpi.MPI.SUM


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

    def __init__(self, *args, **kwargs):

        self._domain = None
        self._base_cell_width = 0.0  # no cell structure imposed

        self._cell_to_particle_map = None
        self._halo_manager = None

        self._position_dat = None

        # Registered particle dats.
        self.particle_dats = []

        # Local number of particles
        self._npart_local = 0

        # Global number of particles
        self._npart = 0

        # number of particles including halos
        self.npart_halo = 0

        self.version_id = 0

        self.invalidate_lists = False
        """If true, all cell lists/ neighbour lists should be rebuilt."""

        self._move_controller = _move_controller(state=self)

        # halo vars
        self._halo_exchange_sizes = None

        self.determine_update_funcs = []
        self.pre_update_funcs = []
        self.post_update_funcs = []

        self._dat_len = 0

    def rebuild_cell_to_particle_maps(self):
        pass

    def cell_decompose(self, cell_width):

        # decompose into larger cells if needed
        if cell_width > self._base_cell_width:

            # decompose the domain and create associated cell list
            assert self._domain is not None, "no domain to decompose"
            assert self._domain.cell_decompose(cell_width) is True, "Requested new cell size cannot be made"
            self._base_cell_width = cell_width
            self._cell_particle_map_setup()
            self.invalidate_lists = True

            for dat in self.particle_dats:
                getattr(self, dat).vid_halo_cell_list = -1

            return True
        else:
            return False

    def _cell_particle_map_setup(self):

        # Can only setup a cell to particle map after a domain and a position
        # dat is set
        if (self._domain is not None) and (self._position_dat is not None):
            # print "setting up cell list"

            self._domain.boundary_condition.set_state(self)

            self._cell_to_particle_map = cell.CellList(
                self.as_func('npart_local'),
                self.get_position_dat(),
                self.domain
            )

            self._cell_to_particle_map.setup_pre_update(self._pre_update_func)
            self._cell_to_particle_map.setup_update_tracking(
                self._determine_update_status
            )
            self._cell_to_particle_map.setup_callback_on_update(
                self._post_update_funcs
            )

            self._cell_to_particle_map.update_required = True

            self._halo_manager = halo.CartesianHaloSix(_AsFunc(self, '_domain'),
                                                       self._cell_to_particle_map)

    def _pre_update_func(self):
        for foo in self.pre_update_funcs:
            foo()

    def _post_update_funcs(self):
        for foo in self.post_update_funcs:
            foo()

    def _determine_update_status(self):
        if len(self.determine_update_funcs) == 0:
            return True

        v = False
        for foo in self.determine_update_funcs:
            v |= foo()
        return v
    
    def _check_comm(self):
        if self._domain is None:
            raise RuntimeError('A domain is required but no domain was found')
        if self._domain.comm is None:
            raise RuntimeError('A domain communicator is required but comm was none')        

    @property
    def domain(self):
        return self._domain

    @property
    def _ccomm(self):
        self._check_comm()
        return self._domain.comm

    @property
    def _ccrank(self):
        self._check_comm()
        return self._domain.comm.Get_rank()

    @property
    def _ccsize(self):
        self._check_comm()
        return self._domain.comm.Get_size()

    def _ccbarrier(self):
        self._check_comm()
        return self._domain.comm.Barrier()

    @domain.setter
    def domain(self, new_domain):
        self._domain = new_domain

        if mpi.decomposition_method == mpi.decomposition.spatial:
            self._domain.mpi_decompose()

        self._cell_particle_map_setup()

    def get_npart_local_func(self):
        return self.as_func('npart_local')

    def get_cell_to_particle_map(self):
        return self._cell_to_particle_map

    def get_position_dat(self):
        assert self._position_dat is not None, "No positions have been added, " \
                                               "Use data.PositionDat"
        return getattr(self, self._position_dat)

    def __setattr__(self, name, value):
        """
        Works the same as the default __setattr__ except that particle dats are registered upon being
        added. Added particle dats are registered in self.particle_dats.

        :param name: Name of parameter.
        :param value: Value of parameter.
        :return:
        """

        # Add to instance list of particle dats.
        if issubclass(type(value), data.ParticleDat):
            object.__setattr__(self, name, value)
            if name not in self.particle_dats:
                self.particle_dats.append(name)
            else:
                raise RuntimeError('This property is already assigned')

            # Recreate the move library.
            self._move_controller.reset()

            # register resize callback
            getattr(self, name)._resize_callback = self._resize_callback

            # add self to dats group
            getattr(self, name).group = self

            # set dat name to be attribute name
            getattr(self, name).name = name
            
            # set the communicator of the dat
            if self._domain is not None:
                getattr(self, name).comm = self._ccomm

            if self._dat_len > 0:
                getattr(self, name).resize(self._dat_len, _callback=False)
            elif self._npart_local > 0:
                getattr(self, name).resize(self._npart_local, _callback=False)
            else:
                getattr(self, name).resize(self._npart, _callback=False)

            if type(value) is data.PositionDat:
                if self._position_dat is None:
                    self._position_dat = name
                    self._cell_particle_map_setup()
                else:
                    raise RuntimeError('A state may hold only 1 PositionDat.')

        elif (type(value) is data.global_array.GlobalArrayClassic) or \
            (type(value) is data.global_array.GlobalArrayShared):
            object.__setattr__(self, name, value)
            if self._domain is not None:
                getattr(self, name).comm = self._ccomm

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
    def npart_local(self):
        """
        :return: Local number of particles
        """
        return self._npart_local

    @npart_local.setter
    def npart_local(self, value):
        """
        Set local number of particles.

        :arg value: New number of local particles.
        """
        self._npart_local = int(value)
        for ix in self.particle_dats:
            _dat = getattr(self, ix)
            _dat.npart_local = int(value)
            _dat.halo_start_reset()

    @property
    def npart(self):
        return self._npart

    @npart.setter
    def npart(self, value=None):
        assert value >= 0, "no value passed"
        self._npart = value

    def _resize_callback(self, value=None):
        """
        Work around until types are implemented. The assumptions for the macro
        based pairlooping assume that all ParticleDats have the halo space
        allocated. This method is registered with ParticleDats added to the
        state such that resizing one dat resizes all dats.
        """
        assert type(value) is not None, "No new size passed"
        self._dat_len = int(value)
        for ix in self.particle_dats:
            _dat = getattr(self, ix)
            _dat.resize(int(value), _callback=False)

    def scatter_data_from(self, rank):
        self.broadcast_data_from(rank)
        self.filter_on_domain_boundary()

    def broadcast_data_from(self, rank=0):
        # This is in terms of MPI_COMM_WORLD
        assert (rank > -1) and (rank < self._ccsize), "Invalid mpi rank"

        if self._ccsize == 1:
            self.npart_local = self.npart
            return
        else:
            s = np.array([self.npart])
            self._ccomm.Bcast(s, root=rank)
            self.npart_local = s[0]
            for px in self.particle_dats:
                getattr(self, px).broadcast_data_from(rank=rank, _resize_callback=False)

    def gather_data_on(self, rank=0):
        # also in terms of MPI_COMM_WORLD
        if self._ccsize == 1:
            return
        else:
            for px in self.particle_dats:
                getattr(self, px).gather_data_on(rank)

    def filter_on_domain_boundary(self):
        """
        Remove particles that do not reside in this subdomain. State requires a
        domain and a PositionDat
        """
        b = self.domain.boundary
        p = self.get_position_dat()

        lo = np.logical_and(
            np.logical_and(
                np.logical_and((b[0] <= p[::, 0]), (p[::, 0] < b[1])),
                np.logical_and((b[2] <= p[::, 1]), (p[::, 1] < b[3]))
            ),
            np.logical_and((b[4] <= p[::, 2]), (p[::, 2] < b[5]))
        )

        bx = np.logical_not(lo)
        self._move_controller.compress_empty_slots(np.nonzero(bx)[0])

        self.npart_local = np.sum(lo)

        # for px in self.particle_dats:
        #    getattr(self, px).npart_local = self.npart_local

        # check we did not loose some particles in the process
        self.check_npart_total()

    def check_npart_total(self):
        """Check no particles have been lost"""
        t = self.sum_npart_local()
        if t != self.npart:
            raise RuntimeError("Particles lost! Expected {} found {}.".
                               format(self.npart, t))

    def sum_npart_local(self):
        """Sum npart_local across all ranks"""
        _t = np.array((0,), dtype=ctypes.c_int64)
        _o = np.array((self.npart_local,), dtype=ctypes.c_int64)
        self._ccomm.Allreduce(_o, _t)
        return _t[0]

    def move_to_neighbour(self, ids_directions_list=None, dir_send_totals=None,
                          shifts=None):
        self._move_controller.move_to_neighbour(
            ids_directions_list,
            dir_send_totals,
            shifts
        )

    def _halo_update_exchange_sizes(self):

        idi = self._cell_to_particle_map.version_id
        idh = self._cell_to_particle_map.halo_version_id
        if idi == 0:
            raise RuntimeError('Cell to particle map was never constructed before' +
                               ' a call to halo exchange')

        if idi > idh:
            self._halo_exchange_sizes = self._halo_manager.exchange_cell_counts()
            new_size = self.npart_local + self._halo_exchange_sizes[0]
            self._resize_callback(new_size)
            self._cell_to_particle_map.prepare_halo_sort(new_size)
            self.npart_halo = new_size - self.npart_local
        return self._halo_exchange_sizes

    def _halo_update_post_exchange(self):
        idi = self._cell_to_particle_map.version_id
        idh = self._cell_to_particle_map.halo_version_id
        if idi > idh:
            self._cell_to_particle_map.post_halo_exchange()


class State(BaseMDState):
    pass


class _move_controller(object):
    def __init__(self, *args, **kwargs):

        self.state = kwargs['state']

        self._move_dir_recv_totals = None
        self._move_dir_send_totals = None

        self._move_shift_array = host.NullDoubleArray

        self._move_send_buffer = None
        self._move_recv_buffer = None

        self._move_unpacking_lib = None
        self._move_packing_lib = None
        self._move_empty_slots = host.Array(ncomp=4, dtype=ctypes.c_int)
        self._move_used_free_slot_count = None

        self._total_ncomp = None

        # Timers
        self.move_timer = ppmd.opt.Timer(runtime.TIMER, 0)

        self._status = mpi.MPI.Status()

        # Timers
        self.move_timer = ppmd.opt.Timer(runtime.TIMER, 0)
        self.compress_timer = ppmd.opt.Timer(runtime.TIMER, 0)

        self._status = mpi.MPI.Status()

        # compressing vars
        self._compressing_lib = None

        self.compressed = True
        """ Bool to determine if the held :class:`~data.ParticleDat` members have gaps in them. """

        self.uncompressed_n = False

    def reset(self):
        # Reset these to ensure that move libs are rebuilt.
        self._move_packing_lib = None
        self._move_unpacking_lib = None
        self._move_send_buffer = None
        self._move_recv_buffer = None

        # Re-create the move library.
        self._total_ncomp = 0
        for ixi, ix in enumerate(self.state.particle_dats):
            _dat = getattr(self.state, ix)
            self._total_ncomp += _dat.ncomp * ctypes.sizeof(_dat.dtype)

    def move_to_neighbour(self, ids_directions_list=None,
                          dir_send_totals=None, shifts=None):
        """
        Move particles using the linked list.
        :arg host.Array ids_directions_list(int): Linked list of ids from
         directions.
        :arg host.Array dir_send_totals(int): 26 Element array of number of
        particles traveling in each direction.
        :arg host.Array shifts(double): 73 element array of the shifts to
        apply when moving particles for the 26 directions.
        """

        self.move_timer.start()

        if self._move_packing_lib is None:
            self._move_packing_lib = _move_controller.build_pack_lib(
                self.state)

        _send_total = dir_send_totals.data.sum()
        # Make/resize send buffer.
        if self._move_send_buffer is None:
            self._move_send_buffer = host.Array(
                ncomp=self._total_ncomp * _send_total, dtype=ctypes.c_byte)

        elif self._move_send_buffer.ncomp < self._total_ncomp * _send_total:
            self._move_send_buffer.realloc(self._total_ncomp * _send_total)

        # Make recv sizes array.
        if self._move_dir_recv_totals is None:
            self._move_dir_recv_totals = host.Array(ncomp=26,
                                                    dtype=ctypes.c_int)

        # exchange number of particles about to be sent.
        self._move_dir_send_totals = dir_send_totals

        self._move_dir_recv_totals.zero()
        self._move_exchange_send_recv_sizes()

        # resize recv buffer.
        _recv_total = self._move_dir_recv_totals.data.sum()

        # using uint_8 in library
        assert ctypes.sizeof(ctypes.c_byte) == 1

        if self._move_recv_buffer is None:
            self._move_recv_buffer = host.Array(
                ncomp=self._total_ncomp * _recv_total, dtype=ctypes.c_byte)

        elif self._move_recv_buffer.ncomp < self._total_ncomp * _recv_total:
            self._move_recv_buffer.realloc(self._total_ncomp * _recv_total)

        for ix in self.state.particle_dats:
            _d = getattr(self.state, ix)
            if _recv_total + self.state.npart_local > _d.max_npart:
                _d.resize(_recv_total + self.state.npart_local)

        # Empty slots store.
        if _send_total > 0:
            self._resize_empty_slot_store(_send_total)

        # pack particles to send.
        assert shifts.dtype == ctypes.c_double
        self._move_packing_lib(
            self._move_send_buffer.ctypes_data,
            shifts.ctypes_data,
            ids_directions_list.ctypes_data,
            self._move_empty_slots.ctypes_data,
            *[getattr(self.state, n).ctypes_data for n in self.state.particle_dats]
        )

        # sort empty slots.
        self._move_empty_slots.data[0:_send_total:].sort()

        # exchange particle data.
        self._exchange_move_send_recv_buffers()

        # Create unpacking lib.
        if self._move_unpacking_lib is None:
            self._move_unpacking_lib = _move_controller.build_unpack_lib(
                self.state)

        # unpack recv buffer.
        self._move_unpacking_lib(
            ctypes.c_int(_recv_total),
            ctypes.c_int(_send_total),
            ctypes.c_int(self.state.npart_local),
            self._move_empty_slots.ctypes_data,
            self._move_recv_buffer.ctypes_data,
            *[getattr(self.state, n).ctypes_data for n in self.state.particle_dats]
        )

        _recv_rank = np.zeros(26)
        _send_rank = np.zeros(26)

        for _tx in range(26):
            direction = mpi.recv_modifiers[_tx]

            _send_rank[_tx] = mpi.cartcomm_shift(
                self._ccomm, direction, ignore_periods=True)

            _recv_rank[_tx] = mpi.cartcomm_shift(
                self._ccomm,
                (-1 * direction[0], -1 * direction[1], -1 * direction[2]),
                ignore_periods=True
            )

        if _recv_total < _send_total:
            self.compressed = False
            _tmp = self._move_empty_slots.data[_recv_total:_send_total:]
            self._move_empty_slots.data[0:_send_total - _recv_total:] = np.array(_tmp, copy=True)

        else:
            self.state.npart_local = self.state.npart_local + _recv_total - _send_total

        # Compress particle dats.
        self._compress_particle_dats(_send_total - _recv_total)

        if _send_total > 0 or _recv_total > 0:
            # print "invalidating lists in move"
            self.state.invalidate_lists = True

        self.move_timer.pause()

        return True

    def _exchange_move_send_recv_buffers(self):

        _s_start = 0
        _s_end = 0
        _r_start = 0
        _r_end = 0
        _n = self._total_ncomp

        for ix in range(26):
            _s_end += _n * self._move_dir_send_totals[ix]
            _r_end += _n * self._move_dir_recv_totals[ix]

            direction = mpi.recv_modifiers[ix]

            _send_rank = mpi.cartcomm_shift(
                self._ccomm,
                direction, ignore_periods=True
            )
            _recv_rank = mpi.cartcomm_shift(
                self._ccomm,
                (-1 * direction[0], -1 * direction[1], -1 * direction[2]),
                ignore_periods=True
            )

            # sending of particles.
            if self._move_dir_send_totals[ix] > 0 and self._move_dir_recv_totals[ix] > 0:
                self._ccomm.Sendrecv(
                    self._move_send_buffer.data[_s_start:_s_end:],
                    _send_rank,
                    _send_rank,
                    self._move_recv_buffer.data[_r_start:_r_end:],
                    _recv_rank,
                    self._ccrank,
                    self._status
                )

            elif self._move_dir_send_totals[ix] > 0:
                    self._ccomm.Send(
                        self._move_send_buffer.data[_s_start:_s_end:],
                        _send_rank,
                        _send_rank
                    )

            elif self._move_dir_recv_totals[ix] > 0:
                    self._ccomm.Recv(
                        self._move_recv_buffer.data[_r_start:_r_end:],
                        _recv_rank,
                        self._ccrank,
                        self._status
                    )

            _s_start += _n * self._move_dir_send_totals[ix]
            _r_start += _n * self._move_dir_recv_totals[ix]

            if self._move_dir_recv_totals[ix] > 0:
                _tsize = self._status.Get_count(mpi.mpi_map[ctypes.c_byte])
                assert _tsize == self._move_dir_recv_totals[ix] * self._total_ncomp, \
                    "RECVD incorrect amount of data:" + str(_tsize) + " " + str(
                    self._move_dir_recv_totals[ix] * self._total_ncomp)

    def _move_exchange_send_recv_sizes(self):
        """
        Exhange the sizes expected in the next particle move.
        """
        _status = mpi.MPI.Status()

        for ix in range(26):

            direction = mpi.recv_modifiers[ix]

            _send_rank = mpi.cartcomm_shift(
                self._ccomm, direction, ignore_periods=True)

            _recv_rank = mpi.cartcomm_shift(
                self._ccomm,
                (-1 * direction[0], -1 * direction[1], -1 * direction[2]),
                ignore_periods=True
            )

            self._ccomm.Sendrecv(
                self._move_dir_send_totals.data[ix:ix + 1:],
                _send_rank,
                _send_rank,
                self._move_dir_recv_totals.data[ix:ix + 1:],
                _recv_rank,
                self._ccrank,
                _status
            )

    def _compress_particle_dats(self, num_slots_to_fill):
        """
        Compress the particle dats held in the state. Compressing removes empty rows.
        """
        _compressing_n_new = host.Array([0], dtype=ctypes.c_int)
        assert self._compressing_lib is not None
        if self.compressed is True:
            return
        else:

            self.compress_timer.start()
            self._compressing_lib(
                ctypes.c_int(num_slots_to_fill),
                ctypes.c_int(self.state.npart_local),
                self._move_empty_slots.ctypes_data,
                _compressing_n_new.ctypes_data,
                *[getattr(self.state, n).ctypes_data for n in self.state.particle_dats]
            )

            self.state.npart_local = _compressing_n_new[0]
            self.compressed = True
            # self._move_empty_slots = []
            self.compress_timer.pause()

    def compress_empty_slots(self, slots):
        if self._compressing_lib is None:
            self._compressing_lib = _move_controller.build_compress_lib(
                self.state)

        le = len(slots)
        if le > 0:
            self.compressed = False
        else:
            self.compressed = True
        if le > 0:
            self._resize_empty_slot_store(le)
            self._move_empty_slots[0:le:] = slots
            self._compress_particle_dats(le)

    def _resize_empty_slot_store(self, new_size):
        new_size = max(new_size, 1)
        if self._move_empty_slots.ncomp < new_size:
            self._move_empty_slots.realloc(new_size)

    @property
    def _ccomm(self):
        return self.state.domain.comm

    @property
    def _ccrank(self):
        return self.state.domain.comm.Get_rank()

    @property
    def _ccsize(self):
        return self.state.domain.comm.Get_size()

    def _ccbarrier(self):
        return self.state.domain.comm.Barrier()

    @staticmethod
    def build_unpack_lib(state):

        dats = state.particle_dats

        def g(x):
            return getattr(state, x)

        args = ','.join(['{} * D_{}'.format(g(n).ctype, n) for n in dats])
        mvs = ''.join([
            '''
               memcpy(&D_{0}[pos * {1}], _R_BUF, {2});
               _R_BUF += {2};
            '''.format(
                str(n),
                str(g(n).ncomp),
                str(g(n).ncomp * ctypes.sizeof(g(n).dtype))
            ) for n in dats
        ])
        hsrc = '''
        #include <string.h>
        #include <stdint.h>
        '''
        src = '''
        extern "C"
        int move_unpack(
            const int _recv_count,
            const int _num_free_slots,
            const int _prev_num_particles,
            const int * _free_slots,
            const uint8_t * _R_BUF,
            %(ARGS)s
        ){
        for(int ix = 0; ix < _recv_count; ix++){
            int pos;
            // prioritise filling spaces in dat.
            if (ix < _num_free_slots) {pos = _free_slots[ix];}
            else {pos = _prev_num_particles + ix - _num_free_slots;}
            %(MVS)s
        }
        return 0;}
        ''' % {'ARGS': args, 'MVS': mvs}
        return build.simple_lib_creator(
            hsrc, src, 'move_unpack')['move_unpack']

    @staticmethod
    def build_pack_lib(state):

        dats = state.particle_dats

        def g(x):
            return getattr(state, x)

        args = ','.join(['const {} * D_{}'.format(g(n).ctype, n) for n in dats])
        _dynamic_dats_shift = ''
        for ix in state.particle_dats:
            dat = g(ix)
            sub_dict = {
                'DTYPE': dat.ctype,
                'DBYTE': str(ctypes.sizeof(dat.dtype)),
                'TBYTE': str(dat.ncomp * ctypes.sizeof(dat.dtype)),
                'NCOMP': str(dat.ncomp),
                'NAME': str(ix)
            }

            if type(dat) is data.PositionDat:
                assert dat.ncomp == 3, "move only defined in 3D"
                _dynamic_dats_shift += '''
                %(DTYPE)s _pos_tmp[3];
                _pos_tmp[0]=D_%(NAME)s[_ix* %(NCOMP)s    ]+SHIFT[(_dir*3)    ];
                _pos_tmp[1]=D_%(NAME)s[_ix* %(NCOMP)s + 1]+SHIFT[(_dir*3) + 1];
                _pos_tmp[2]=D_%(NAME)s[_ix* %(NCOMP)s + 2]+SHIFT[(_dir*3) + 2];
                memcpy(_S_BUF, _pos_tmp, 3*%(DBYTE)s);
                _S_BUF += %(TBYTE)s;
                ''' % sub_dict
            else:
                assert dat.ncomp > 0, "move not defined for 0 component dats"
                _dynamic_dats_shift += '''
                memcpy(_S_BUF, &D_%(NAME)s[_ix * %(NCOMP)s], %(TBYTE)s);
                _S_BUF += %(TBYTE)s;
                ''' % sub_dict

        hsrc = '''
        #include<string.h>
        #include<stdint.h>
        '''
        src = '''
        extern "C"
        int move_pack(
        uint8_t * _S_BUF,
        const double * SHIFT,
        const int * _direction_id_list,
        int * _empty_slot_store,
        %(ARGS)s
        ){
            // Next free space in send buffer.
            int _slot_index = 0;
            //loop over the send directions.
            for(int _dir = 0; _dir < 26; _dir++){
                //traverse linked list.
                int _ixd = _direction_id_list[_dir];
                while(_ixd > -1){
                    //Generate code based on ParticleDats
                    int _ix = _direction_id_list[_ixd];
                    \n%(DYNAMIC_DATS)s
                    _empty_slot_store[_slot_index] = _ix;
                    _slot_index += 1;
                    _ixd = _direction_id_list[_ixd+1];
                }
            }
        return 0;}''' % {'DYNAMIC_DATS': _dynamic_dats_shift, 'ARGS': args}
        return build.simple_lib_creator(hsrc, src, 'move_pack')['move_pack']

    @staticmethod
    def build_compress_lib(state):

        dats = state.particle_dats

        def g(x):
            return getattr(state, x)

        hsrc = ''''''
        args = ','.join(['{} * D_{}'.format(g(n).ctype, n) for n in dats])
        dyn = '\n'.join(
            ['''
             for(int ix=0 ; ix<{0} ; ix++){{
             {1}[dest*{0}+ix] = {1}[src*{0}+ix];}}
             '''.format(
                str(g(n).ncomp),
                'D_{}'.format(n)
            ) for n in dats]
        )

        src = '''
        extern "C"
        int compress(
            const int slots_to_fill_in,
            const int n_new_in,
            const int * slots,
            int * n_new_out,
            %(ARGS)s
        ){
        int slots_to_fill = slots_to_fill_in;
        int n_new = n_new_in;
        int last_slot;
        int last_slot_lookup_index = slots_to_fill - 1;
        int dest_index = 0;
        int dest = -1;
        // Whilst there are slots to fill and the current slot is not past the
        // end of the array.
        if (n_new > 0) {
            while ( (dest_index <= last_slot_lookup_index) &&
             (slots[dest_index] < n_new) ){
                // get first empty slot in particle dats.
                dest = slots[dest_index];
                int src = -1;
                //loop from end to empty slot
                for (int iy = n_new - 1; iy > dest; iy--){
                    if (iy == slots[last_slot_lookup_index]){
                        n_new = iy;
                        last_slot_lookup_index--;
                        //printf("n_new=%%d \\n", n_new);
                    } else {
                        src = iy;
                        break;
                    }
                }
                if (src > 0){
                    \n%(DYN_DAT_CODE)s
                    n_new = src;
                } else {
                    n_new = slots[last_slot_lookup_index];
                    break;
                }
                dest_index++;
            }
        }
        n_new_out[0] = n_new;
        return 0;}
        ''' % {'DYN_DAT_CODE': dyn, 'ARGS': args}

        return build.simple_lib_creator(hsrc, src, 'compress')['compress']







