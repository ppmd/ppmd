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


        # move vars.
        self._move_ncomp = []
        self._move_total_ncomp = 0
        self._move_packing_lib = None
        self._move_send_buffer = None
        self._move_recv_buffer = None

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

    def move_to_neighbour(self, ids=None, direction=None):
        """
        Move particles to a neighbouring process.
        :return: bool as to whether move was successful.
        """

        assert direction is not None, "move_to_neighbour error: No direction passed."
        assert type(direction) is tuple, "move_to_neighbour error: passed direction should be a tuple."

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

        # Get number of components that will need to be packed/unpacked.
        if self._move_send_buffer is None or self._move_recv_buffer is None:
            self._move_ncomp = []
            self._move_total_ncomp = 0
            for ix in self.particle_dats:
                _dat = getattr(self, ix)
                self._move_ncomp.append(_dat.ncomp)
                self._move_total_ncomp += _dat.ncomp


        # Create a send buffer.
        if self._move_send_buffer is None:
            self._move_send_buffer = host.Array(ncomp=_send_count * self._move_total_ncomp)


        elif _send_count * self._move_total_ncomp > self._move_send_buffer.ncomp:
            if runtime.VERBOSE.level > 2:
                print "rank", mpi.MPI_HANDLE.rank, ": move send buffer resized."

            self._move_send_buffer.realloc(_send_count * self._move_total_ncomp)


        if self._move_packing_lib is None:

            _dynamic_dats = ''
            _space = ' ' * 16

            for ix, iy in zip(self.particle_dats, self._move_ncomp):

                # make case where ParticleDat has more than one component.
                if iy > 1:
                    _dynamic_dats += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                    _dynamic_dats += _space + 'SEND_BUFFER[index+ni] = %(NAME)s[(_ix*%(NCOMP)s)+ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                    _dynamic_dats += _space + '} \n'
                    _dynamic_dats += _space + 'index += %(NCOMP)s; \n' % {'NCOMP':iy}
                else:
                    _dynamic_dats += _space + 'SEND_BUFFER[index] = %(NAME)s[_ix]; \n' % {'NAME':str(ix)}
                    _dynamic_dats += _space + 'index += 1; \n'

            _packing_code = '''
            int index = 0;
            for (int _ix = 0; _ix < end; _ix++){
                \n%(DYNAMIC_DATS)s
            }

            ''' % {'DYNAMIC_DATS': _dynamic_dats}

            _packing_headers = ['stdio.h']
            _packing_static_args = {'end':ctypes.c_int}
            _packing_args = {'SEND_BUFFER':self._move_send_buffer}

            # Dynamic arguments dependant on how many particle dats there are.
            for idx, ix in enumerate(self.particle_dats):
                # existing dat in state
                _packing_args['%(NAME)s' % {'NAME':ix}] = getattr(self, ix)

            # create a unique but searchable name.
            _name = ''
            for ix in self.particle_dats:
                _name += '_' + str(ix)

            # make kernel
            _packing_kernel = kernel.Kernel('state_move_packing' + _name, _packing_code, None, _packing_headers, None, _packing_static_args)

            # make packing library
            self._move_packing_lib = build.SharedLib(_packing_kernel, _packing_args)

        # Execute library.
        self._move_packing_lib.execute(static_args={'end': ctypes.c_int(len(ids))})





















        return True






