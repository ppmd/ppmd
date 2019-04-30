from __future__ import print_function, division, absolute_import

# system level
import ctypes
import numpy as np
import math

# package level
import ppmd.mpi
import ppmd.state
import ppmd.kernel
import ppmd.access
import ppmd.pio
import ppmd.host


from ppmd.state_modifier import StateModifier, StateModifierContext

from ppmd import data

# cuda level
from ppmd.cuda import cuda_cell, cuda_halo, cuda_data, cuda_runtime, \
    cuda_mpi, cuda_loop, cuda_build, cuda_base

_AsFunc = ppmd.state._AsFunc
_MPI = ppmd.mpi.MPI
SUM = _MPI.SUM
_MPIWORLD = ppmd.mpi.MPI.COMM_WORLD
_MPIRANK = ppmd.mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = ppmd.mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = ppmd.mpi.MPI.COMM_WORLD.Barrier

class BaseMDState(object):
    """
    Create an empty state to which particle properties such as position, velocities are added as
    attributes.
    """

    def __init__(self):

        self._domain = None


        self._cell_to_particle_map = cuda_cell.CellOccupancyMatrix()

        self._halo_manager = None
        self._halo_device_version = -1

        self._halo_sizes = None
        self._halo_cell_max_b = 0
        self._halo_cell_max_h = 0

        self._halo_h_scan = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)
        self._halo_b_scan = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)

        self._halo_h_groups_se_indices = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)
        self._halo_b_groups_se_indices = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)

        self._halo_h_cell_indices = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)
        self._halo_b_cell_indices = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)

        self._halo_h_cell_counts = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)
        self._halo_b_cell_counts = cuda_base.Array(ncomp=1, dtype=ctypes.c_int32)

        self._halo_send_counts = ppmd.host.Array(ncomp=6, dtype=ctypes.c_int32)
        self._halo_tmp_space = cuda_base.Array(ncomp=10, dtype=ctypes.c_double)
        self._halo_position_shifts = cuda_base.Array(ncomp=18, dtype=ctypes.c_double)


        self._position_dat = None

        # Registered particle dats.
        self.particle_dats = []

        # Local number of particles
        self._npart_local = 0

        # Global number of particles
        self._npart = 0

        # do the ParticleDats have gaps in them?
        self.compressed = True
        """ Bool to determine if the held :class:`~cuda_data.ParticleDat` members have gaps in them. """

        self.uncompressed_n = False


        # compression vars
        self._filter_method = None
        self._comp_replacement_find_method = _FindCompressionIndices()
        self._compression_lib = None

        # State version id
        self.version_id = 0

        # move vars
        self._move_send_ranks = None
        self._move_recv_ranks = None
        self._move_send_buffer = None
        self._move_recv_buffer = None
        self._move_lib = None
        self._move_send_counts = None
        self._move_recv_counts = None
        self._empty_per_particle_flag = None

        # move vars.

        self.invalidate_lists = False
        """If true, all cell lists/ neighbour lists should be rebuilt."""
        self.determine_update_funcs = []
        self.pre_update_funcs = []
        self.post_update_funcs = []
        self._gdm = None

        self._state_modifier_context = StateModifierContext(self)
        self.modifier = StateModifier(self)

    def modify(self):
        return self._state_modifier_context


    def check_position_consistency(self):
        if self._gdm is not None:
            self._gdm()
        else:
            raise RuntimeError('Cannot check particle decomposition, was a domain added?.')

    @property
    def _ccomm(self):
        return self._domain.comm
    @property
    def _ccrank(self):
        return self._domain.comm.Get_rank()
    @property
    def _ccsize(self):
        return self._domain.comm.Get_size()
    def _ccbarrier(self):
        return self._domain.comm.Barrier()
    def rebuild_cell_to_particle_maps(self):
        pass

    def _halo_update_exchange_sizes(self):
        idi = self._cell_to_particle_map.version_id
        idh = self._cell_to_particle_map.version_id_halo

        if idi > idh:
            # update boundary and halo cell layout if domain has changed
            if self.domain.cell_array.version > self._halo_device_version:
                 self._halo_update_groups()
            # update boundary and halo cell counts and exchange these
            self._halo_sizes = self._halo_manager.exchange_cell_counts()

            # scan of ccc of boundary cells.
            self._halo_b_scan.realloc(self._halo_b_cell_indices.ncomp+1)
            self._halo_cell_max_b = cuda_mpi.cuda_exclusive_scan_int_masked_copy(
                length=self._halo_b_cell_indices.ncomp,
                d_map=self._halo_b_cell_indices,
                d_ccc=self._cell_to_particle_map.cell_contents_count,
                d_scan=self._halo_b_scan
            )

            # scan of ccc of halo cells.
            self._halo_h_scan.realloc(self._halo_h_cell_indices.ncomp+1)
            self._halo_cell_max_h = cuda_mpi.cuda_exclusive_scan_int_masked_copy(
                length=self._halo_h_cell_indices.ncomp,
                d_map=self._halo_h_cell_indices,
                d_ccc=self._cell_to_particle_map.cell_contents_count,
                d_scan=self._halo_h_scan
            )

            # update the cell occupancy matrix to assign correct particle indices
            # to correct cell and layer
            self._halo_update_cell_to_particle_map()


            # update array of send counts.

            # print self._halo_manager.get_boundary_cell_groups()[1][:]
            cuda_halo.update_send_counts(self._halo_manager.get_boundary_cell_groups()[1],
                                         self._halo_b_scan,
                                         self._halo_send_counts)

            # resize tmp space
            s = max(self._move_ncomp)
            if self._halo_tmp_space.ncomp < (s * self._halo_sizes[1]):
                self._halo_tmp_space.realloc(s * 2 * self._halo_sizes[1])



    def _halo_update_cell_to_particle_map(self):

        self._cell_to_particle_map.prepare_halo_sort(self._halo_cell_max_h)

        cuda_halo.update_cell_occ_matrix(self._halo_h_cell_indices.ncomp,
                                         self._halo_cell_max_h,
                                         self._cell_to_particle_map.layers_per_cell,
                                         self.npart_local,
                                         self._halo_h_cell_indices,
                                         self._cell_to_particle_map.cell_contents_count,
                                         self._halo_h_scan,
                                         self._cell_to_particle_map.matrix)

        self._cell_to_particle_map.version_id_halo += 1





    def _halo_update_groups(self):
        hm = self._halo_manager
        hmb = hm.get_boundary_cell_groups()
        self._halo_b_cell_indices.realloc(hmb[0].ncomp)
        cuda_runtime.cuda_mem_cpy(d_ptr=self._halo_b_cell_indices.ctypes_data,
                                  s_ptr=hmb[0].ctypes_data,
                                  size=ctypes.c_size_t(hmb[0].ncomp * ctypes.sizeof(ctypes.c_int)),
                                  cpy_type="cudaMemcpyHostToDevice")

        self._halo_b_groups_se_indices.realloc(hmb[1].ncomp)
        cuda_runtime.cuda_mem_cpy(d_ptr=self._halo_b_groups_se_indices.ctypes_data,
                                  s_ptr=hmb[1].ctypes_data,
                                  size=ctypes.c_size_t(hmb[1].ncomp * ctypes.sizeof(ctypes.c_int)),
                                  cpy_type="cudaMemcpyHostToDevice")

        hmh = hm.get_halo_cell_groups()

        self._halo_h_cell_indices.realloc(hmh[0].ncomp)

        cuda_runtime.cuda_mem_cpy(d_ptr=self._halo_h_cell_indices.ctypes_data,
                                  s_ptr=hmh[0].ctypes_data,
                                  size=ctypes.c_size_t(hmh[0].ncomp * ctypes.sizeof(ctypes.c_int)),
                                  cpy_type="cudaMemcpyHostToDevice")

        self._halo_h_groups_se_indices.realloc(hmh[1].ncomp)
        cuda_runtime.cuda_mem_cpy(d_ptr=self._halo_h_groups_se_indices.ctypes_data,
                                  s_ptr=hmh[1].ctypes_data,
                                  size=ctypes.c_size_t(hmh[1].ncomp * ctypes.sizeof(ctypes.c_int)),
                                  cpy_type="cudaMemcpyHostToDevice")

        self._halo_position_shifts[:] = hm.get_position_shifts()[:]

        self._halo_device_version = self.domain.cell_array.version


    def _cell_particle_map_setup(self):

        # Can only setup a cell to particle map after a domain and a position
        # dat is set
        if (self._domain is not None) and (self._position_dat is not None):

            if self._domain.boundary_condition is not None:
                self._domain.boundary_condition.set_state(self)

            #print "setting up cell list"
            self._cell_to_particle_map.setup(self.as_func('npart_local'),
                                             self.get_position_dat(),
                                             self.domain)
            self._cell_to_particle_map.trigger_update()

            self._cell_to_particle_map.setup_pre_update(self._pre_update_func)
            self._cell_to_particle_map.setup_update_tracking(
                self._determine_update_status
            )
            self._cell_to_particle_map.setup_callback_on_update(
               self._post_update_funcs
            )


            # initialise the filter method now we have a domain and positions
            self._filter_method = _FilterOnDomain(self,
                                                  self._domain,
                                                  self.get_position_dat())
            if self._ccsize == 1:
                self._halo_manager = cuda_halo.CartesianHalo(self._cell_to_particle_map)
            else:
                self._halo_manager = ppmd.halo.CartesianHaloSix(_AsFunc(self, '_domain'),
                                                                self._cell_to_particle_map)

    def _pre_update_func(self):
        for foo in self.pre_update_funcs:
            foo()

    def _post_update_funcs(self):
        for foo in self.post_update_funcs:
            foo()
    def _determine_update_status(self):
        v = False
        for foo in self.determine_update_funcs:
            v |= foo()
        return v

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, new_domain):
        self._domain = new_domain

        if ppmd.mpi.decomposition_method == ppmd.mpi.decomposition.spatial:
            self._domain.mpi_decompose()

        self._cell_particle_map_setup()
        self._gdm = data.data_movement.GlobalDataMover(self)

    def get_npart_local_func(self):
        return self.as_func('npart_local')

    def get_cell_to_particle_map(self):
        return self._cell_to_particle_map


    def get_position_dat(self):
        assert self._position_dat is not None, "No positions have been added, " \
                                               "Use cuda_data.PositionDat"
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
        if issubclass(type(value), cuda_data.ParticleDat):

            object.__setattr__(self, name, value)
            self.particle_dats.append(name)

            # Reset these to ensure that move libs are rebuilt.
            self._move_packing_lib = None
            self._move_send_buffer = None
            self._move_recv_buffer = None

            # force rebuild of compression lib
            self._compression_lib = None

            # Re-create the move library.
            self._total_ncomp = 0
            self._move_ncomp = []
            for ixi, ix in enumerate(self.particle_dats):
                _dat = getattr(self, ix)
                self._move_ncomp.append(_dat.ncomp)
                self._total_ncomp += _dat.ncomp

            # register resize callback
            getattr(self, name)._resize_callback = self._resize_callback

            # add self to dats group
            getattr(self, name).group = self

            # set dat name to be attribute name
            getattr(self, name).name = name


            if self._npart_local > 0:
                getattr(self, name).resize(self._npart_local, _callback=False)
            else:
                getattr(self, name).resize(self._npart, _callback=False)

            getattr(self, name).npart_local = self._npart_local

            if type(value) is cuda_data.PositionDat:
                self._position_dat = name
                self._cell_particle_map_setup()

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
        # resize dats if needed
        self._resize_callback(value)

        self._npart_local = int(value)
        for ix in self.particle_dats:
            _dat = getattr(self,ix)
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
        for ix in self.particle_dats:
            _dat = getattr(self,ix)
            _dat.resize(int(value), _callback=False)

    def scatter_data_from(self, rank):
        self.broadcast_data_from(rank)
        self.filter_on_domain_boundary()


    def broadcast_data_from(self, rank=0):
        # uses MPI_COMM_WORLD
        assert (rank>-1) and (rank<_MPISIZE), "Invalid mpi rank"

        if _MPISIZE == 1:
            self.npart_local = self.npart
            return
        else:
            s = np.array([self.get_position_dat().nrow])
            _MPIWORLD.Bcast(s, root=rank)
            self.npart_local = s[0]
            for px in self.particle_dats:
                getattr(self, px).broadcast_data_from(rank=rank, _resize_callback=False)

    def gather_data_on(self, rank=0):
        # uses MPI_COMM_WORLD
        if _MPISIZE == 1:
            return
        else:
            for px in self.particle_dats:
                getattr(self, px).gather_data_on(rank)

    def free_all(self):
        for px in self.particle_dats:
            getattr(self, px).free()

    def filter_on_domain_boundary(self):
        """
        Remove particles that do not reside in this subdomain. State requires a
        domain and a PositionDat
        """
        self._compress_dats(*self._filter_method.apply())


    def remove_by_slot(self, slots):

        remover = _RemoveBySlot(self, slots)
        self._compress_dats(*remover.apply())



    def _compress_dats(self,
                       empty_per_particle_flag,
                       empty_slots,
                       num_slots_to_fill,
                       new_npart):
        """
        empty_per_particle_flag takes the form of an exlusive sum array
        """


        replacement_slots = \
            self._comp_replacement_find_method.apply(
                empty_per_particle_flag,
                num_slots_to_fill,
                new_npart,
                self.npart_local-new_npart
            )


        if self._compression_lib is None:
            self._compression_lib = _CompressParticleDats(
                self, self.particle_dats
            )



        if replacement_slots is not None:
            self._compression_lib.apply(num_slots_to_fill,
                                        empty_slots,
                                        replacement_slots)



        self.npart_local = new_npart




    def move_to_neighbour(self, directions_matrix=None, dir_counts=None):
        """
        Move particles using the passed matrix where rows correspond to
        directions.
        """

        if self._move_lib is None:
            self._move_lib = \
                cuda_build.build_static_libs('cudaMoveLib')

        self._move_send_ranks, self._move_recv_ranks = \
            ppmd.mpi.cartcomm_get_move_send_recv_ranks(self._ccomm)

        self._move_send_ranks = ppmd.host.Array(initial_value=self._move_send_ranks,
                                                dtype=ctypes.c_int32)
        self._move_recv_ranks = ppmd.host.Array(initial_value=self._move_recv_ranks,
                                                dtype=ctypes.c_int32)
        self._move_recv_counts = ppmd.host.Array(ncomp=26,
                                                 dtype=ctypes.c_int32)
        self._move_send_counts = ppmd.host.Array(initial_value=dir_counts[:],
                                                 dtype=ctypes.c_int32)


        ndats = len(self.particle_dats)
        ptr_t = ndats*ctypes.c_void_p
        byte_t = ndats*ctypes.c_int32
        ptrs_a = []
        byte_a = []
        total_bytes = 0

        for dat in self.particle_dats:
            dath = getattr(self, dat)
            ptrs_a.append(dath.ctypes_data)
            be = ctypes.sizeof(dath.dtype)*dath.ncomp
            byte_a.append(be)
            total_bytes += be

        # These are arrays len=ndat, of dat pointers and dat byte counts per
        # particle
        ptrs = ptr_t(*ptrs_a)
        byte = byte_t(*byte_a)


        cuda_mpi.cuda_mpi_err_check(
            self._move_lib['cudaMoveStageOne'](
                ctypes.c_int32(self._ccomm.py2f()),
                self._move_send_ranks.ctypes_data,
                self._move_recv_ranks.ctypes_data,
                self._move_send_counts.ctypes_data,
                self._move_recv_counts.ctypes_data
            )
        )


        total_particles = np.sum(dir_counts[:])
        tl = total_particles*total_bytes

        if self._move_send_buffer is None:
            self._move_send_buffer = cuda_base.Array(ncomp=tl,
                                                     dtype=ctypes.c_int8)
        elif self._move_send_buffer.ncomp < tl:
            self._move_send_buffer.realloc_zeros(tl)


        # resize tmp buffers
        total_recv_count = np.sum(self._move_recv_counts[:])*total_bytes
        recv_count = np.sum(self._move_recv_counts[:])

        if self._move_recv_buffer is None:
            self._move_recv_buffer = cuda_base.Array(ncomp=total_recv_count,
                                                     dtype=ctypes.c_int8)
        elif self._move_recv_buffer.ncomp < total_recv_count:
            self._move_recv_buffer.realloc_zeros(total_recv_count)


        # resize dats
        new_ncomp = self.get_position_dat().npart_local + recv_count
        if self._empty_per_particle_flag is None:
            self._empty_per_particle_flag = cuda_base.Array(
                ncomp=new_ncomp,
                dtype=ctypes.c_int32
            )
        elif self._empty_per_particle_flag.ncomp < new_ncomp:
            self._empty_per_particle_flag.realloc_zeros(new_ncomp)
        else:
            self._empty_per_particle_flag.zero()


        self._resize_callback(self.npart_local + recv_count)


        # pack -> S/R unpack

        #print ppmd.mpi.MPI_HANDLE.rank, self.domain.boundary[:]
        #print self.npart_local, total_particles, recv_count

        cuda_mpi.cuda_mpi_err_check(
            self._move_lib['cudaMoveStageTwo'](
                ctypes.c_int32(self._ccomm.py2f()),
                ctypes.c_int32(self.npart_local),
                ctypes.c_int32(total_bytes),
                ctypes.c_int32(ndats),
                self._move_send_counts.ctypes_data,
                self._move_recv_counts.ctypes_data,
                self._move_send_ranks.ctypes_data,
                self._move_recv_ranks.ctypes_data,
                directions_matrix.ctypes_data,
                ctypes.c_int32(directions_matrix.ncol),
                self._move_send_buffer.ctypes_data,
                self._move_recv_buffer.ctypes_data,
                ctypes.byref(ptrs),
                ctypes.byref(byte),
                self._empty_per_particle_flag.ctypes_data
            )
        )

        self.npart_local = self.npart_local + recv_count



    def _move_build_unpacking_lib(self):
        pass


    def _exchange_move_send_recv_buffers(self):
        pass


    def _move_exchange_send_recv_sizes(self):
        """
        Exhange the sizes expected in the next particle move.
        """
        pass

    def _move_build_packing_lib(self):
        """
        Build the library to pack particles to send.
        """
        pass


    def _compress_particle_dats(self, num_slots_to_fill):
        """
        Compress the particle dats held in the state. Compressing removes empty rows.
        """
        pass


class State(BaseMDState):
    pass




class _BaseRemover:

    def __init__(self, state):

        self._empty_slots = cuda_data.ScalarArray(dtype=ctypes.c_int)
        self._replacement_slots = None
        self.state = state
        self._per_particle_flag = cuda_data.ParticleDat(ncomp=1, dtype=ctypes.c_int)
        self._find_indices_method = _FindCompressionIndices()

    def _specific_method(self):
        pass

    def apply(self):

        n = self.state.npart_local
        self._per_particle_flag.resize(n+1)
        self._per_particle_flag.npart_local = n
        self._per_particle_flag.zero()
        
        self._specific_method()

        # exclusive scan on array of flags
        cuda_runtime.LIB_CUDA_MISC['cudaExclusiveScanInt'](
                                   self._per_particle_flag.ctypes_data,
                                   ctypes.c_int(n+1))

        # number leaving is in element n+1
        end_ptr = ppmd.host.pointer_offset(self._per_particle_flag.ctypes_data,
                                           n*ctypes.sizeof(ctypes.c_int))

        n2_ = ctypes.c_int()
        cuda_runtime.cuda_mem_cpy(ctypes.byref(n2_),
                                  end_ptr,
                                  ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
                                  'cudaMemcpyDeviceToHost')
        n2 = n2_.value

        # compute new npart_local
        new_n = n-n2

        # the empty slots before the new end need filling
        end_ptr = ppmd.host.pointer_offset(self._per_particle_flag.ctypes_data,
                                           new_n*ctypes.sizeof(ctypes.c_int))
        n_to_fill_ = ctypes.c_int()
        cuda_runtime.cuda_mem_cpy(ctypes.byref(n_to_fill_),
                                  end_ptr,
                                  ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
                                  'cudaMemcpyDeviceToHost')

        # number to fill in [0, npart_local - 1]
        n_to_fill = n_to_fill_.value

        # if there are empty slots
        if n2 > 0:
            self._empty_slots.resize(n_to_fill)
            self._empty_slots.zero()

            args = list(cuda_runtime.kernel_launch_args_1d(new_n, threads=1024)) + \
                   [self._per_particle_flag.ctypes_data,
                    ctypes.c_int(new_n),
                    self._empty_slots.ctypes_data]

            cuda_runtime.cuda_err_check(cuda_mpi.LIB_CUDA_MPI['cudaFindEmptySlots'](
                *args
                )
            )

        # this first returned array actaully is an exclusive sum of the flags
        return self._per_particle_flag, self._empty_slots, n_to_fill, new_n



class _RemoveBySlot(_BaseRemover):

    def __init__(self, state, slots_to_remove):
        super().__init__(state)

        self._slots = slots_to_remove

    def _specific_method(self):
        
        with self._per_particle_flag.modify_view() as m:
            m[self._slots, 0] = 1



class _FilterOnDomain(_BaseRemover):
    """
    Use a domain boundary and a PositionDat to construct and return:
    A new number of local particles
    A number of empty slots
    A list of empty slots
    A list of particles to move into the corresponding empty slots

    The output can then be used to compress ParticleDats.
    """

    def __init__(self, state, domain_in, positions_in):

        super().__init__(state)

        self._domain = domain_in
        self._positions = positions_in

        kernel1_code = """
        int _F = 0;
        if ((P(0) < B0) ||  (P(0) >= B1)) { _F = 1; }
        if ((P(1) < B2) ||  (P(1) >= B3)) { _F = 1; }
        if ((P(2) < B4) ||  (P(2) >= B5)) { _F = 1; }
        F(0) = _F;
        """

        kernel1_dict = {
            'P': self._positions(ppmd.access.R),
            'F': self._per_particle_flag(ppmd.access.W)
        }

        kernel1_statics = {
            'B0': ctypes.c_double,
            'B1': ctypes.c_double,
            'B2': ctypes.c_double,
            'B3': ctypes.c_double,
            'B4': ctypes.c_double,
            'B5': ctypes.c_double
        }

        kernel1 = ppmd.kernel.Kernel('FilterOnDomainK1',
                                     kernel1_code,
                                     None,
                                     static_args=kernel1_statics)

        self._loop1 = cuda_loop.ParticleLoop(kernel1, kernel1_dict)

    def _specific_method(self):

        B = self._domain.boundary

        kernel1_statics = {
            'B0': ctypes.c_double(B[0]),
            'B1': ctypes.c_double(B[1]),
            'B2': ctypes.c_double(B[2]),
            'B3': ctypes.c_double(B[3]),
            'B4': ctypes.c_double(B[4]),
            'B5': ctypes.c_double(B[5])
        }

        # find particles to remove
        self._loop1.execute(n=self._positions.group.npart_local,
                            static_args=kernel1_statics)


class _FindCompressionIndices(object):
    def __init__(self):
        self._replacement_slots = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_int)

    def apply(self, per_particle_flag, n_to_fill, new_n, search_n):
        if n_to_fill == 0:
            return None


        self._replacement_slots.resize(n_to_fill)
        self._replacement_slots.zero()

        args = list(cuda_runtime.kernel_launch_args_1d(search_n, threads=1024)) + \
               [per_particle_flag.ctypes_data,
                ctypes.c_int(new_n),
                ctypes.c_int(search_n),
                self._replacement_slots.ctypes_data]


        cuda_runtime.cuda_err_check(
            cuda_mpi.LIB_CUDA_MPI['cudaFindNewSlots'](*args)
        )


        return self._replacement_slots


class _CompressParticleDats(object):
    def __init__(self, state, particle_dat_names):
        self._state = state
        self._names = particle_dat_names


        call_template = '''
        bs; bs.x = %(NCOMP)s * blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
        ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];
        compression_kernel<%(DTYPE)s><<<bs,ts>>>(%(PTR_NAME)s, %(NCOMP)s);
        '''

        extra_params = ''
        kernel_calls = '''
        '''

        for ix in self._names:
            dat = getattr(self._state, ix)
            sdtype = ppmd.host.ctypes_map[dat.dtype]

            extra_params += ', ' + sdtype + '* ' + ix
            kernel_calls += call_template % {'DTYPE': sdtype,
                                             'PTR_NAME': ix,
                                             'NCOMP': dat.ncomp}


        name = 'compression_lib'

        header_code = '''
        #include "cuda_generic.h"

        __constant__ int d_n_empty;
        __constant__ int* d_e_slots;
        __constant__ int* d_r_slots;

        template <typename T>
        __global__ void compression_kernel(T* __restrict__ d_ptr,
                                           const int ncomp){

            const int ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (ix < d_n_empty * ncomp){

                const int sx = ix / ncomp;
                const int comp = ix %% ncomp;
                const int eslot = d_e_slots[sx];
                const int rslot = d_r_slots[sx];

                d_ptr[eslot*ncomp + comp] = d_ptr[rslot*ncomp + comp];

            }
            return;
        }

        extern "C" int compression_lib(const int blocksize[3],
                                       const int threadsize[3],
                                       const int h_n_empty,
                                       const int* d_e_slots_p,
                                       const int* d_r_slots_p
                                       %(EXTRA_PARAMS)s
                                       );

        ''' % {'EXTRA_PARAMS': extra_params}

        src_code = '''

        int compression_lib(const int blocksize[3],
                            const int threadsize[3],
                            const int h_n_empty,
                            const int* d_e_slots_p,
                            const int* d_r_slots_p
                            %(EXTRA_PARAMS)s
                            ){

            dim3 bs;
            dim3 ts;
            checkCudaErrors(cudaMemcpyToSymbol(d_n_empty, &h_n_empty, sizeof(int)));
            checkCudaErrors(cudaMemcpyToSymbol(d_e_slots, &d_e_slots_p, sizeof(int*)));
            checkCudaErrors(cudaMemcpyToSymbol(d_r_slots, &d_r_slots_p, sizeof(int*)));


            %(KERNEL_CALLS)s


            return (int) cudaDeviceSynchronize();

        }
        ''' % {'KERNEL_CALLS': kernel_calls, 'EXTRA_PARAMS': extra_params}


        self._lib = cuda_build.simple_lib_creator(header_code, src_code, name)[name]


    def apply(self, n_to_fill, empty_slots, replacement_slots):

        args = list(cuda_runtime.kernel_launch_args_1d(n_to_fill, threads=1024))
        args.append(ctypes.c_int(n_to_fill))
        args.append(empty_slots.ctypes_data)
        args.append(replacement_slots.ctypes_data)

        for ix in self._names:
            args.append(getattr(self._state, ix).ctypes_data)

        cuda_runtime.cuda_err_check(self._lib(*args))


class _SortParticlesOnCell(object):
    def __init__(self, state, tmp_space):
        self._state = state
        self._tmp_space  = tmp_space

        dat_max_bytes = 0

        for ixi, ix in enumerate(self._state.particle_dats):
            dat = getattr(self._state, ix)
            dat_max_bytes = max(dat_max_bytes,
                int(
                    dat.ncomp*ctypes.sizeof(dat.dtype)
                )
            )

        self._max_bytes = dat_max_bytes
        self._lib = None


    def apply(self):

        occ_matrix = self._state.get_cell_to_particle_map()
        ccc_scan = occ_matrix.cell_contents_count_scan()
        crl_array = occ_matrix.cell_reverse_lookup
        pl_array = occ_matrix.particle_layers

        bytes_needed = self._state.npart_local * self._max_bytes

        if self._tmp_space.ncomp * ctypes.sizeof(self._tmp_space.dtype) < bytes_needed:
            new_ncomp = math.ceil(float(bytes_needed)/ctypes.sizeof(self._tmp_space.dtype))
            self._tmp_space.realloc_zeros(new_ncomp)

        if self._lib is None:
            self._build_lib()


    def _build_lib(self):


        call_template = '''
        bs; bs.x = %(NCOMP)s * blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
        ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];
        copy_to_tmp<%(DTYPE)s><<<bs,ts>>>(%(PTR_NAME)s, d_tmp, %(NCOMP)s);

        err = (int) cudaDeviceSynchronize();
        if (err > 0) {
            cout << "Error: copy_to_tmp %(PTR_NAME)s << endl;
            return err;
        }

        copy_from_tmp<%(DTYPE)s><<<bs,ts>>>(%(PTR_NAME)s, d_tmp, %(NCOMP)s);

        err = (int) cudaDeviceSynchronize();
        if (err > 0) {
            cout << "Error: copy_from_tmp %(PTR_NAME)s << endl;
            return err;
        }

        '''

        extra_params = ''
        kernel_calls = '''
        '''

        for ix in self._state.particle_dats:
            dat = getattr(self._state, ix)
            sdtype = ppmd.host.ctypes_map[dat.dtype]

            extra_params += ', ' + sdtype + '* ' + ix
            kernel_calls += call_template % {'DTYPE': sdtype,
                                             'PTR_NAME': ix,
                                             'NCOMP': dat.ncomp}


        name = 'SortParticlesOnCellLib'

        header_code = '''
        #include "cuda_generic.h"

        __constant__ int d_npart_local;
        __constant__ int *d_ccc_scan;
        __constant__ int *d_crl_array;
        __constant__ int *d_pl_array;


        // ----------------------------------------------------

        template <typename T>
        __global__ void copy_to_tmp(
            const T* __restrict__ d_ptr,
            T* __restrict__ d_tmp,
            const int ncomp
        ){

            const int ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (ix < d_npart_local * ncomp){

                d_tmp[ix] = d_ptr[ix];

            }
            return;
        }

        // ----------------------------------------------------

        template <typename T>
        __global__ void copy_from_tmp(
            T* __restrict__ d_ptr,
            const T* __restrict__ d_tmp,
            const int ncomp
        ){

            const int ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (ix < d_npart_local * ncomp){

                const int sx = ix / ncomp;
                const int comp = ix %% ncomp;

                // get cell index
                const int cx = d_crl_array[sx];
                // get cell layer
                const int cl = d_pl_array[sx];
                // calculate new index
                const int new_slot = d_ccc_scan[cx] + cl;

                d_ptr[new_slot*ncomp + comp] = d_tmp[sx*ncomp + comp];

            }
            return;
        }

        // ----------------------------------------------------

        __global__ void correct_occ_matrix(
            const int d_num_layers,
            int * __restrict__ d_occ_matrix
        ){
            const int ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (ix < d_npart_local){
                // get cell index
                const int cx = d_crl_array[sx];
                // get cell layer
                const int cl = d_pl_array[sx];

                d_occ_matrix[cx*d_num_layers + cl] = ix;

            }

            return;
        }


        // ----------------------------------------------------


        extern "C" int %(NAME)s(const int blocksize[3],
                                const int threadsize[3],
                                const int h_npart_local,
                                const int* d_ccc_scanp,
                                const int* d_crl_arrayp,
                                const int* d_pl_arrayp,
                                int *d_tmp,
                                int *d_occ_matrix,
                                const int h_num_layers,
                                %(EXTRA_PARAMS)s
        );

        ''' % {'EXTRA_PARAMS': extra_params, 'NAME':name}


        src_code = '''

        int %(NAME)s(const int blocksize[3],
                     const int threadsize[3],
                     const int h_npart_local,
                     const int* d_ccc_scanp,
                     const int* d_crl_arrayp,
                     const int* d_pl_arrayp,
                     int *d_tmp,
                     int *d_occ_matrix,
                     const int h_num_layers,
                     %(EXTRA_PARAMS)s
        ){
            int err = 0;

            dim3 bs;
            dim3 ts;
            checkCudaErrors(cudaMemcpyToSymbol(d_npart_local, &h_npart_local, sizeof(int)));
            checkCudaErrors(cudaMemcpyToSymbol(d_ccc_scan, &d_ccc_scanp, sizeof(int*)));
            checkCudaErrors(cudaMemcpyToSymbol(d_crl_array, &d_crl_arrayp, sizeof(int*)));
            checkCudaErrors(cudaMemcpyToSymbol(d_pl_array, &d_pl_arrayp, sizeof(int*)));


            %(KERNEL_CALLS)s


            bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];
            correct_occ_matrix<<<bs,ts>>>(h_num_layers, d_occ_matrix);


            err = (int) cudaDeviceSynchronize();
            if (err > 0) {
                cout << "Error: correct_occ_matrix << endl;
                return err;
            }


            return err;

        }
        ''' % {'KERNEL_CALLS': kernel_calls,
               'EXTRA_PARAMS': extra_params,
               'NAME': name}

























