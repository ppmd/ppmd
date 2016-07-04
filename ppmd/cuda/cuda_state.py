
# system level
import ctypes
import numpy as np

# package level
import ppmd.mpi
import ppmd.state

# import build
# import data
# import cell
# import host
# import kernel
# import runtime
# import halo

# cuda level
import cuda_cell
import cuda_halo
import cuda_data
import cuda_runtime

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

        # State time
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
        if issubclass(type(value), cuda_data.ParticleDat):
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


    def filter_on_domain_boundary(self):
        """
        Remove particles that do not reside in this subdomain. State requires a
        domain and a PositionDat
        """
        pass


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




