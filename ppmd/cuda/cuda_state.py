
# system level
import ctypes
import numpy as np

# package level
import ppmd.mpi
import ppmd.state
import ppmd.kernel
import ppmd.access
import ppmd.pio
import ppmd.host


# cuda level
import cuda_cell
import cuda_halo
import cuda_data
import cuda_runtime
import cuda_mpi
import cuda_loop
import cuda_build

_AsFunc = ppmd.state._AsFunc



class BaseMDState(object):
    """
    Create an empty state to which particle properties such as position, velocities are added as
    attributes.
    """

    def __init__(self):


        self._domain = None


        self._cell_to_particle_map = cuda_cell.CellOccupancyMatrix()

        self._halo_manager = cuda_halo.CartesianHalo(self._cell_to_particle_map)

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
        self._empty_per_particle_flag = None
        self._empty_slots = None
        self._replacement_slots = None
        self._new_npart = None
        self._num_slots_to_fill = None
        self._compression_lib = None

        # State version id
        self.version_id = 0


        # move vars.

        self.invalidate_lists = False
        """If true, all cell lists/ neighbour lists should be rebuilt."""



    def _cell_particle_map_setup(self):

        # Can only setup a cell to particle map after a domain and a position
        # dat is set
        if (self._domain is not None) and (self._position_dat is not None):
            #print "setting up cell list"
            self._cell_to_particle_map.setup(self.as_func('npart_local'),
                                             self.get_position_dat(),
                                             self.domain)
            self._cell_to_particle_map.trigger_update()

            # initialise the filter method now we have a domain and positions
            self._filter_method = _FilterOnDomain(self._domain,
                                                  self.get_position_dat())


    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, new_domain):
        self._domain = new_domain

        print "before decomp"
        if ppmd.mpi.decomposition_method == ppmd.mpi.decomposition.spatial:
            self._domain.mpi_decompose()

        cuda_runtime.cuda_device_reset()
        cuda_runtime.cuda_set_device()

        print "after decomp"

        self._cell_particle_map_setup()

    def get_npart_local_func(self):
        return self.as_func('npart_local')

    def get_domain(self):
        return self._domain

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
        if (issubclass(type(value), cuda_data.ParticleDat) and not name.startswith('_') ):
            object.__setattr__(self, name, value)
            self.particle_dats.append(name)

            # Reset these to ensure that move libs are rebuilt.
            self._move_packing_lib = None
            self._move_send_buffer = None
            self._move_recv_buffer = None

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

            # resize to Ntotal for time being
            getattr(self, name).resize(self._npart, _callback=False)


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
        self._npart_local = int(value)
        for ix in self.particle_dats:
            _dat = getattr(self,ix)
            _dat.npart_local = int(value)
            _dat.halo_start_reset()
        # print "N set:", value


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
        assert (rank>-1) and (rank<ppmd.mpi.MPI_HANDLE.nproc), "Invalid mpi rank"

        if ppmd.mpi.MPI_HANDLE.nproc == 1:
            self.npart_local = self.npart
            return
        else:
            s = np.array([self.get_position_dat().nrow])
            ppmd.mpi.MPI_HANDLE.comm.Bcast(s, root=rank)
            self.npart_local = s[0]
            for px in self.particle_dats:
                getattr(self, px).broadcast_data_from(rank=rank, _resize_callback=False)

    def gather_data_on(self, rank=0):
        if ppmd.mpi.MPI_HANDLE.nproc == 1:
            return
        else:
            for px in self.particle_dats:
                getattr(self, px).gather_data_on(rank)


    def filter_on_domain_boundary(self):
        """
        Remove particles that do not reside in this subdomain. State requires a
        domain and a PositionDat
        """

        self._empty_per_particle_flag, \
        self._empty_slots, \
        self._num_slots_to_fill, \
        self._new_npart = self._filter_method.apply()



        n = self.npart
        n2 = n - self._new_npart
        new_n = self._new_npart

        test = np.zeros([n+1, 2])
        test[:,0] = self._empty_per_particle_flag[:,0]

        ppmd.pio.rprint('\n', test)
        ppmd.pio.rprint('\n new_n, n2 ', new_n, " " , n2)
        ppmd.pio.rprint(n - self._empty_per_particle_flag[n,0])

        ppmd.pio.rprint('\n E:', self._empty_slots[:])


        self._replacement_slots = \
            self._comp_replacement_find_method.apply(self._empty_per_particle_flag,
                                                     self._num_slots_to_fill,
                                                     self._new_npart,
                                                     self.npart-self._new_npart)

        ppmd.pio.rprint('\n R:', self._replacement_slots[:])

        self._compression_lib = _CompressParticleDats(self, self.particle_dats)
        self._compression_lib.apply(self._num_slots_to_fill,
                                    self._empty_slots,
                                    self._replacement_slots)


        self.npart_local = self._new_npart






    def move_to_neighbour(self, ids_directions_list=None, dir_send_totals=None, shifts=None):
        """
        Move particles using the linked list.

        :arg host.Array ids_directions_list(int): Linked list of ids from directions.
        :arg host.Array dir_send_totals(int): 26 Element array of number of particles traveling in each direction.
        :arg host.Array shifts(double): 73 element array of the shifts to apply when moving particles for the 26 directions.
        """
        pass


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




class _FilterOnDomain(object):
    """
    Use a domain boundary and a PositionDat to construct and return:
    A new number of local particles
    A number of empty slots
    A list of empty slots
    A list of particles to move into the corresponding empty slots

    The output can then be used to compress ParticleDats.
    """

    def __init__(self, domain_in, positions_in):
        self._domain = domain_in
        self._positions = positions_in


        self._empty_slots = cuda_data.ScalarArray(dtype=ctypes.c_int)
        self._replacement_slots = None
        self._per_particle_flag = cuda_data.ParticleDat(ncomp=1, dtype=ctypes.c_int)

        kernel1_code = """

        int _F = 0;

        if ((P(0) < B0) ||  (P(0) >= B1)) {
            _F = 1;
        }
        if ((P(1) < B2) ||  (P(1) >= B3)) {
            _F = 1;
        }
        if ((P(2) < B4) ||  (P(2) >= B5)) {
            _F = 1;
        }
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

        self._find_indices_method = _FindCompressionIndices()


    def apply(self):

        n = self._positions.group.npart

        self._per_particle_flag.resize(n+1)
        B = self._domain.boundary

        kernel1_statics = {
            'B0': ctypes.c_double(B[0]),
            'B1': ctypes.c_double(B[1]),
            'B2': ctypes.c_double(B[2]),
            'B3': ctypes.c_double(B[3]),
            'B4': ctypes.c_double(B[4]),
            'B5': ctypes.c_double(B[5])
        }


        self._per_particle_flag.zero()

        print "nlocal", self._per_particle_flag.npart, self._per_particle_flag.ncol,self._positions.group.npart

        # find particles to remove
        self._loop1.execute(n=self._positions.group.npart,
                            static_args=kernel1_statics)





        # exclusive scan on array of flags
        cuda_runtime.LIB_CUDA_MISC['cudaExclusiveScanInt'](
                                   self._per_particle_flag.ctypes_data,
                                   ctypes.c_int(n+1))



        end_ptr = ppmd.host.pointer_offset(self._per_particle_flag.ctypes_data,
                                           n*ctypes.sizeof(ctypes.c_int))


        n2_ = ctypes.c_int()
        cuda_runtime.cuda_mem_cpy(ctypes.byref(n2_),
                                  end_ptr,
                                  ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
                                  'cudaMemcpyDeviceToHost')
        n2 = n2_.value
        new_n = n-n2


        end_ptr = ppmd.host.pointer_offset(self._per_particle_flag.ctypes_data,
                                           new_n*ctypes.sizeof(ctypes.c_int))
        n_to_fill_ = ctypes.c_int()
        cuda_runtime.cuda_mem_cpy(ctypes.byref(n_to_fill_),
                                  end_ptr,
                                  ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
                                  'cudaMemcpyDeviceToHost')
        n_to_fill = n_to_fill_.value


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



        return self._per_particle_flag, self._empty_slots, n_to_fill, new_n



class _FindCompressionIndices(object):
    def __init__(self):
        self._replacement_slots = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_int)

    def apply(self, per_particle_flag, n_to_fill, new_n, search_n):

        self._replacement_slots.resize(n_to_fill)
        self._replacement_slots.zero()

        args = list(cuda_runtime.kernel_launch_args_1d(search_n, threads=1024)) + \
               [per_particle_flag.ctypes_data,
                ctypes.c_int(new_n),
                ctypes.c_int(search_n),
                self._replacement_slots.ctypes_data]

        cuda_runtime.cuda_err_check(cuda_mpi.LIB_CUDA_MPI['cudaFindNewSlots'](
            *args
            )
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




