


# system level
import ctypes

# package level
import ppmd.kernel

# cuda level
import cuda_loop
import cuda_data






class FilterOnDomain(object):
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



        self._new_npart_local = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_int)
        self._empty_slot_count = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_int)
        self._empty_slots = cuda_data.ScalarArray(dtype=ctypes.c_int)
        self._replacement_slots = cuda_data.ScalarArray(dtype=ctypes.c_int)
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
        F[0] = _F;

        """

        kernel1_dict = {
            'P': self._positions,
            'F': self._per_particle_flag
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


    def apply(self):

        self._per_particle_flag.resize(self._positions.npart_local+1)
        B = self._domain.boundary

        kernel1_statics = {
            'B0': ctypes.c_double(B[0]),
            'B1': ctypes.c_double(B[1]),
            'B2': ctypes.c_double(B[2]),
            'B3': ctypes.c_double(B[3]),
            'B4': ctypes.c_double(B[4]),
            'B5': ctypes.c_double(B[5])
        }

        self._loop1.execute(n=self._positions.group.npart_local,
                            static_args=kernel1_statics)


        return self._new_npart_local, \
               self._empty_slot_count, \
               self._empty_slots, \
               self._replacement_slots














