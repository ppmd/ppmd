import numpy as np
import particle
import math
import ctypes
import time
import random
import pairloop
import data
import kernel
import loop
import build

np.set_printoptions(threshold='nan')


################################################################################################################
# BaseMDState DEFINITIONS
################################################################################################################

class BaseMDState(object):
    """
    Base molecular dynamics class.
    
    :arg domain domain: Container within which the simulation takes place.
    :arg potential potential: Potential to use between particles.
    :arg PosInit* particle_pos_init: Class to initialise particles with.
    :arg VelInit* particle_vel_init: Class to initialise particles velocities with.
    :arg int n: Number of particles, default 1.
    :arg double mass: Mass of particles, default 1.0.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, domain, potential, particle_pos_init=None, particle_vel_init=None, particle_mass_init=None, n=0):
        """
        Initialise class to hold the state of a simulation.
        :arg domain domain: Container within which the simulation takes place.
        :arg potential potential: Potential to use between particles.
        :arg int n: Number of particles, default 1.
        :arg double mass: Mass of particles, default 1.0        
        
        """

        self._potential = potential
        self._N = n
        self._NT = n
        self._NH = 0  # number in halo
        self._pos = particle.Dat(n, 3, name='positions')
        self._vel = particle.Dat(n, 3, name='velocities')
        self._accel = particle.Dat(n, 3, name='accelerations')

        '''Store global ids of particles'''
        self._global_ids = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)

        '''Lookup table between id and particle type'''
        self._types = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)

        '''Mass is an example of a property dependant on particle type'''
        self._mass = particle.TypedDat(self._NT, 1, 1.0)

        self._domain = domain
        self._kinetic_energy_loop = None

        # potential energy, kinetic energy, total energy.
        self._U = data.ScalarArray(max_size=2, name='potential_energy')
        self._U.InitHaloDat()

        self._K = data.ScalarArray()
        self._Q = data.ScalarArray()

        '''Get domain extent from position config'''
        particle_pos_init.get_extent(self)

        '''Attempt to initialise cell array'''
        self._cell_setup_attempt = self._domain.set_cell_array_radius(self._potential._rn)

        ''' Initialise particle positions'''
        particle_pos_init.reset(self)

        ''' Initialise state time to zero'''
        self._time = 0.0

        '''Setup boundaries'''
        self._domain.bc_setup(self)

        '''Initialise velocities'''
        if particle_vel_init is not None:
            particle_vel_init.reset(self)

        '''Initialise masses'''
        if particle_mass_init is not None:
            particle_mass_init.reset(self)

        # Setup acceleration updating from given potential
        _potential_dat_dict = self._potential.datdict(self)

        if self._cell_setup_attempt is True:
            self._cell_sort_setup()

            self._domain.bc_execute()
            self._cell_sort_all()

            self._looping_method_accel = pairloop.PairLoopRapaportHalo(n=self.n,
                                                                       domain=self._domain,
                                                                       positions=self._pos,
                                                                       potential=self._potential,
                                                                       dat_dict=_potential_dat_dict,
                                                                       cell_list=self._q_list)

        else:
            self._looping_method_accel = pairloop.DoubleAllParticleLoopPBC(n=self.n,
                                                                           domain=self._domain,
                                                                           kernel=self._potential.kernel,
                                                                           particle_dat_dict=_potential_dat_dict)

        self.timer = build.Timer(build.TIMER, 0)

    @property
    def time(self):
        """
        Returns the time of the system.
        """
        return self._time

    def add_time(self, increment=0.0):
        """
        Increases the state time by the increment amount.
        :param increment: Amount to increment time by.
        :return:
        """
        self._time += increment

    @property
    def mpi_handle(self):
        return self._domain.mpi_handle

    def _cell_sort_setup(self):
        """
        Creates looping for cell list creation
        """

        '''Construct initial cell list'''
        self._q_list = data.ScalarArray(dtype=ctypes.c_int,
                                        max_size=self._NT * self._domain.cell_count + self._domain.cell_count + 1)

        '''Keep track of number of particles per cell'''
        self._cell_contents_count = data.ScalarArray(np.zeros([self._domain.cell_count], dtype=ctypes.c_int, order='C'),
                                                     dtype=ctypes.c_int)


        # temporary method for index awareness inside kernel.
        self._internal_index = data.ScalarArray(dtype=ctypes.c_int)
        self._internal_N = data.ScalarArray(dtype=ctypes.c_int)

        self._cell_sort_code = '''
        
        //printf("cell start");
        
        const double R0 = P[0]+0.5*E[0];
        const double R1 = P[1]+0.5*E[1];
        const double R2 = P[2]+0.5*E[2];
        
        const int C0 = (int)(R0/CEL[0]);
        const int C1 = (int)(R1/CEL[1]);
        const int C2 = (int)(R2/CEL[2]);
        
        const int val = (C2*CA[1] + C1)*CA[0] + C0;
        
        //printf("val=%d",val);
        
        CCC[val]++;
        
        q[I[0]] = q[n[0] + val];
        q[n[0] + val] = I[0];
        I[0]++;
        '''
        self._cell_sort_dict = {'E': self._domain.extent,
                                'P': self._pos,
                                'CEL': self._domain.cell_edge_lengths,
                                'CA': self._domain.cell_array,
                                'q': self._q_list,
                                'CCC': self._cell_contents_count,
                                'I': self._internal_index,
                                'n': self._internal_N}

        self._cell_sort_kernel = kernel.Kernel('cell_list_method', self._cell_sort_code, headers=['stdio.h'])
        self._cell_sort_loop = loop.SingleParticleLoop(None, None, self._cell_sort_kernel, self._cell_sort_dict)

    def _cell_sort_all(self):
        """
        Construct neighbour list, assigning *all* atoms to cells. Using Rapaport algorithm.
        """
        self._q_list.resize(self._pos.npart + self._pos.npart_halo + self._domain.cell_count + 1)
        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count
        self._internal_N[0] = self._q_list[self._q_list.end]

        self._q_list.dat[self._q_list[self._q_list.end]:self._q_list.end:] = ctypes.c_int(-1)

        self._internal_index[0] = 0
        self._cell_contents_count.zero()
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._cell_sort_loop.execute(start=0, end=self._pos.npart + self._pos.npart_halo,
                                     dat_dict={'E': self._domain.extent,
                                               'P': self._pos,
                                               'CEL': self._domain.cell_edge_lengths,
                                               'CA': self._domain.cell_array,
                                               'q': self._q_list,
                                               'CCC': self._cell_contents_count,
                                               'I': self._internal_index,
                                               'n': self._internal_N})

    def _cell_sort_local(self):
        """
        Construct neighbour list, assigning *local* atoms to cells. Using Rapaport algorithm.
        """

        self._q_list.resize(self._pos.npart + self._pos.npart_halo + self._domain.cell_count + 1)
        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count

        self._internal_N[0] = self._q_list[self._q_list.end]
        self._q_list.dat[self._q_list[self._q_list.end]:self._q_list.end:] = ctypes.c_int(-1)
        self._internal_index[0] = 0

        self._cell_contents_count.zero()
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._cell_sort_loop.execute(start=0,
                                     end=self._pos.npart,
                                     dat_dict={'E': self._domain.extent,
                                               'P': self._pos,
                                               'CEL': self._domain.cell_edge_lengths,
                                               'CA': self._domain.cell_array,
                                               'q': self._q_list,
                                               'CCC': self._cell_contents_count,
                                               'I': self._internal_index,
                                               'n': self._internal_N}
                                     )

    def _cell_sort_halo(self):
        """
        Construct neighbour list, assigning *halo* atoms to cells. Using Rapaport algorithm. Depreciated, use halo sorting methods within the halo class.
        """

        self._q_list.resize(self._pos.npart + self._pos.npart_halo + self._domain.cell_count + 1)
        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count
        self._internal_N[0] = self._q_list[self._q_list.end]

        self._internal_index[0] = self._pos.npart
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._cell_sort_loop.execute(start=self._pos.npart,
                                     end=self._pos.npart + self._pos.npart_halo,
                                     dat_dict={'E': self._domain.extent,
                                               'P': self._pos,
                                               'CEL': self._domain.cell_edge_lengths,
                                               'CA': self._domain.cell_array,
                                               'q': self._q_list,
                                               'CCC': self._cell_contents_count,
                                               'I': self._internal_index,
                                               'n': self._internal_N}
                                     )

    def forces_update(self):
        """
        Updates forces dats using given looping method.
        """

        self.timer.start()

        self._cell_sort_local()

        if self._cell_setup_attempt == True:
            self._domain.halos.set_position_info(self._cell_contents_count, self._q_list)
            self._domain.halos.exchange(self._pos)

        '''
        print "================================"
        print self._domain.rank, self._q_list[0:4:], self._q_list[self._q_list[self._q_list.end]::]
        print "--------------------------------"
        print self._pos[0:4,::]
        print "================================"
        '''

        self.set_forces(ctypes.c_double(0.0))
        self.reset_u()

        if self._N > 0:
            self._looping_method_accel.execute(n=self._q_list[self._q_list.end])

        self.timer.pause()

    def kinetic_energy_update(self):
        """
        Method to update the recorded kinetic energy of the system.
        :return: New kinetic energy.
        """

        if self._kinetic_energy_loop is None:
            _K_kernel_code = '''

            k[0] += (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];

            '''
            _constants_K = []
            _K_kernel = kernel.Kernel('K_kernel', _K_kernel_code, _constants_K)
            self._kinetic_energy_loop = loop.SingleAllParticleLoop(self.n,
                                                                   self._types,
                                                                   _K_kernel,
                                                                   {'V': self._vel, 'k': self._K, 'M': self._mass})
        self._K.scale(0.0)
        self._kinetic_energy_loop.execute()

        return self._K[0]

    def types_map(self):
        """
        Returns the arrays needed by methods to map between particle global ids and particle types.
        """
        return self._types

    @property
    def types(self):
        return self._types

    @property
    def global_ids(self):
        return self._global_ids

    # @property
    def nt(self):
        return self._NT

    def n(self):
        """
        Returns number of particles.
        """
        return self._N

    def set_n(self, val):
        """
        Set number of particles.
        """

        self._N = val

        self._pos.npart = val
        self._pos.halo_start_reset()

        self._vel.npart = val
        self._vel.halo_start_reset()

        self._accel.npart = val
        self._accel.halo_start_reset()

        self._global_ids.ncomp = val
        # self._global_ids.halo_start_reset()

    def nh(self):
        """Return number of particles in halo"""
        return self._pos.npart_halo

    @property
    def domain(self):
        """
        Return the domain used by the state.
        """
        return self._domain

    @property
    def positions(self):
        """
        Return all particle positions.
        """
        return self._pos

    @property
    def velocities(self):
        """
        Return all particle velocities.
        """
        return self._vel

    @property
    def forces(self):
        """
        Return all particle forces.
        """
        return self._accel

    @property
    def masses(self):
        """
        Return all particle masses.
        """
        return self._mass

    def set_forces(self, val):
        """
        Set all forces to given value.
        
        :arg double val: value to set to.
        """

        # self._accel.set_val(val)
        self._accel.dat[0:self._accel.npart:, ::] = val

    @property
    def potential(self):
        return self._potential

    @property
    def u(self):
        """
        Return potential energy
        """
        return self._U

    @property
    def k(self):
        """
        Return Kenetic energy
        """
        return self._K

    @property
    def q(self):
        """
        Return Total energy
        """
        return self._Q

    def reset_u(self):
        """
        Reset potential energy to 0.0
        """
        # self._U._Dat = np.zeros([1], dtype=ctypes.c_double, order='C')
        self._U.scale(0.)

    def u_set(self, u_in):
        """
        Set a kenetic energy value.
        """
        self._U = u_in

    def energy_kinetic(self):
        """
        Calculate and return kinetic energy.
        """

        def squared_sum(x):
            return (x[0] ** 2) + (x[1] ** 2) + (x[2] ** 2)

        print self._N

        energy = 0.
        for i in range(self._N):
            energy += 0.5 * self._mass[i] * squared_sum(self._vel[i, :])

        return float(energy)


################################################################################################################
# BaseMDStatehalo DEFINITIONS
################################################################################################################


class BaseMDStateHalo(BaseMDState):
    """
    Base molecular dynamics class.
    
    :arg domain domain: Container within which the simulation takes place.
    :arg potential potential: Potential to use between particles.
    :arg PosInit* particle_pos_init: Class to initialise particles with.
    :arg VelInit* particle_vel_init: Class to initialise particles velocities with.
    :arg int n: Number of particles, default 1.
    :arg double mass: Mass of particles, default 1.0.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, domain, potential, particle_pos_init=None, particle_vel_init=None, particle_mass_init=None, n=0):
        """
        Initialise class to hold the state of a simulation.
        :arg domain domain: Container within which the simulation takes place.
        :arg potential potential: Potential to use between particles.
        :arg int n: Number of particles, default 1.
        :arg double mass: Mass of particles, default 1.0        
        
        """
        self._potential = potential
        self._N = n
        self._NT = n
        self._pos = particle.Dat(self._NT, 3, name='positions')
        self._vel = particle.Dat(self._NT, 3, name='velocities')
        self._accel = particle.Dat(self._NT, 3, name='accelerations')

        '''Store global ids of particles'''
        self._global_ids = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)

        '''Lookup table between id and particle type'''
        self._types = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)

        '''Mass is an example of a property dependant on particle type'''
        self._mass = particle.TypedDat(self._NT, 1, 1.0)

        self._domain = domain

        # potential energy, kenetic energy, total energy.
        self._U = data.ScalarArray(max_size=2, name='potential_energy')
        self._U.init_halo_dat()

        self._K = data.ScalarArray()
        self._Q = data.ScalarArray()

        '''Get domain extent from position config'''
        particle_pos_init.get_extent(self)

        '''Attempt to initialise cell array'''
        self._cell_setup_attempt = self._domain.set_cell_array_radius(self._potential._rn)

        ''' Initialise particle positions'''
        particle_pos_init.reset(self)

        '''Initialise velocities'''
        if particle_vel_init is not None:
            particle_vel_init.reset(self)

        '''Initialise masses'''
        if particle_mass_init is not None:
            particle_mass_init.reset(self)

        ''' Initialise state time to zero'''
        self._time = 0.0

        self._domain.bc_setup(self)
        self._domain.bc_execute()

        self._kinetic_energy_loop = None

        # Setup acceleration updating from given potential
        _potential_dat_dict = self._potential.datdict(self)

        if self._cell_setup_attempt is True:
            self._cell_sort_setup()

            self._cell_sort_local()

            self._looping_method_accel = pairloop.PairLoopRapaportHalo(n=self._N,
                                                                       domain=self._domain,
                                                                       positions=self._pos,
                                                                       potential=self._potential,
                                                                       dat_dict=_potential_dat_dict,
                                                                       cell_list=self._q_list)

        else:
            self._looping_method_accel = pairloop.DoubleAllParticleLoopPBC(n=self._N,
                                                                           domain=self._domain,
                                                                           kernel=self._potential.kernel,
                                                                           particle_dat_dict=_potential_dat_dict)

        self.timer = build.Timer(build.TIMER, 0)

        if build.DEBUG.level > 0:
            print "DEBUG IS ON"

    def _cell_sort_setup(self):
        """
        Creates looping for cell list creation
        """

        '''Construct initial cell list'''
        self._q_list = data.ScalarArray(dtype=ctypes.c_int,
                                        max_size=self._NT * self._domain.cell_count + 2 * self._domain.cell_count)

        '''Keep track of number of particles per cell'''
        self._cell_contents_count = data.ScalarArray(np.zeros([self._domain.cell_count], dtype=ctypes.c_int, order='C'),
                                                     dtype=ctypes.c_int)

        # temporary method for index awareness inside kernel.
        self._internal_index = data.ScalarArray(dtype=ctypes.c_int)
        self._internal_N = data.ScalarArray(dtype=ctypes.c_int)

        self._cell_sort_code = '''
        
        //printf("start P[0] = %f, P[1] = %f, P[2] = %f |", P[0], P[1], P[2]);
        
        
        const int C0 = (int)((P[0] - B[0])/CEL[0]);
        const int C1 = (int)((P[1] - B[2])/CEL[1]);
        const int C2 = (int)((P[2] - B[4])/CEL[2]);
        
        const int val = (C2*CA[1] + C1)*CA[0] + C0;
        
        //printf("val = %d |", val);
        
        //needed, may improve halo exchange times
        CCC[val]++;

        q[I[0]] = q[n[0] + val];
        
        //printf("I[0] = %d, n[0] = %d, n[0] + val = %d |", I[0], n[0], n[0] + val);
        
        q[n[0] + val] = I[0];

        I[0]++;

        //printf("I[0] = %d |", I[0]);
        //printf("end #");
        '''
        self._cell_sort_dict = {'B': self._domain.boundary_outer,
                                'P': self._pos,
                                'CEL': self._domain.cell_edge_lengths,
                                'CA': self._domain.cell_array,
                                'q': self._q_list,
                                'CCC': self._cell_contents_count,
                                'I': self._internal_index,
                                'n': self._internal_N}

        self._cell_sort_kernel = kernel.Kernel('cell_list_method', self._cell_sort_code, headers=['stdio.h'])
        self._cell_sort_loop = loop.SingleParticleLoop(None, self.types_map, self._cell_sort_kernel,
                                                       self._cell_sort_dict)

    def _cell_sort_local(self):
        """
        Construct neighbour list, assigning *local* atoms to cells. Using Rapaport algorithm.
        """

        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count

        self._internal_N[0] = self._q_list[self._q_list.end]
        self._q_list.dat[self._q_list[self._q_list.end]:self._q_list.end:] = ctypes.c_int(-1)
        self._internal_index[0] = 0

        self._cell_contents_count.zero()
        self._cell_sort_loop.execute(start=0,
                                     end=self._N,
                                     dat_dict={'B': self._domain.boundary_outer,
                                               'P': self._pos,
                                               'CEL': self._domain.cell_edge_lengths,
                                               'CA': self._domain.cell_array,
                                               'q': self._q_list,
                                               'CCC': self._cell_contents_count,
                                               'I': self._internal_index,
                                               'n': self._internal_N}
                                     )

        ################################################################################################################


# PosInitLatticeNRho DEFINITIONS
################################################################################################################        


class PosInitLatticeNRho(object):
    """
    Arrange n particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass.
    
    :arg int n: n, number of particles.
    :arg double rho: :math:`rho`, required density.
    :arg double lx: domain side length, overrides density.
    """

    def __init__(self, n, rho, lx=None):
        """
        Initialise required lattice with the number of particles and required density.
        
       
        """
        self._in_lx = lx
        self._N = n
        self._rho = rho

    def get_extent(self, state_input):
        """
        Initialise domain extents prior to setting particle positions.
        """
        if self._in_lx is None:
            Lx = (float(self._N) / float(self._rho)) ** (1. / 3.)
        else:
            Lx = self._in_lx

        state_input.domain.set_extent(np.array([Lx, Lx, Lx]))

    def reset(self, state_input):
        """
        Applies initial lattice to particle positions.
        
        :arg state state_input: object of state class. Inheritered from BaseMDState.
        """

        # Evaluate cube side length.
        if self._in_lx is None:
            Lx = (float(self._N) / float(self._rho)) ** (1. / 3.)
        else:
            Lx = self._in_lx

        # Cube dimensions of data
        np1_3 = math.ceil(self._N ** (1. / 3.))
        np2_3 = np1_3 ** 2.

        # starting point for each dimension.
        mLx_2 = (-0.5 * Lx) + (0.5 * Lx) / np1_3

        # set new domain extents, see get_extent()
        # state_input.domain.set_extent(np.array([Lx, Lx, Lx]))

        # get pointer for positions
        _p = state_input.positions
        _d = state_input.domain.boundary
        _gid = state_input.global_ids

        # Loop over all particles
        _n = 0
        for ix in range(self._N):

            # Map point into cube side of calculated side length Lx.
            z = math.floor(ix / np2_3)

            _tx = mLx_2 + (math.fmod((ix - z * np2_3), np1_3) / np1_3) * Lx  # x
            _ty = mLx_2 + (math.floor((ix - z * np2_3) / np1_3) / np1_3) * Lx  # y
            _tz = mLx_2 + (z / np1_3) * Lx

            if (_d[0] <= _tx < _d[1]) and (_d[2] <= _ty < _d[3]) and (_d[4] <= _tz < _d[5]):
                _p[_n, 0] = _tx
                _p[_n, 1] = _ty
                _p[_n, 2] = _tz
                _gid[_n] = ix
                _n += 1

        state_input.set_n(_n)
        _p.halo_start_reset()


################################################################################################################
# PosInitLatticeNRhoRand DEFINITIONS random.uniform(0,self._dev)
################################################################################################################          

class PosInitLatticeNRhoRand(object):
    """
    Arrange n particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass adds uniform deviantion based on given maximum.
    
    :arg int n: number of particles.
    :arg double rho: :math:`/rho`, required density.
    :arg double dev: maximum possible random deviation (uniform) from lattice.
    :arg double lx: domain side length, overrides density.    
    """

    def __init__(self, n, rho, dev, lx=None):
        """
        Initialise required lattice with the number of particles and required density.
        
        """
        self._in_lx = lx
        self._N = n
        self._rho = rho
        self._dev = dev

    def get_extent(self, state_input):
        """
        Initialise domain extents prior to setting particle positions.
        """
        if self._in_lx is None:
            Lx = (float(self._N) / float(self._rho)) ** (1. / 3.)
        else:
            Lx = self._in_lx

        state_input.domain.set_extent(np.array([Lx, Lx, Lx]))

    def reset(self, state_input):
        """
        Applies initial lattice to particle positions.
        
        :arg state state_input: object of state class. Inheritered from BaseMDState.
        """

        # Evaluate cube side length.
        if self._in_lx is None:
            Lx = (float(self._N) / float(self._rho)) ** (1. / 3.)
        else:
            Lx = self._in_lx

        # Cube dimensions of data
        np1_3 = math.ceil(self._N ** (1. / 3.))
        np2_3 = np1_3 ** 2.

        # starting point for each dimension.
        mLx_2 = (-0.5 * Lx) + (0.5 * Lx) / np1_3

        # set new domain extents, see get_extent()
        # state_input.domain.set_extent(np.array([Lx, Lx, Lx]))

        # get pointer for positions
        _p = state_input.positions
        _d = state_input.domain.boundary
        _gid = state_input.global_ids

        # Loop over all particles
        _n = 0
        for ix in range(self._N):

            # Map point into cube side of calculated side length Lx.
            z = math.floor(ix / np2_3)

            _tx = mLx_2 + (math.fmod((ix - z * np2_3), np1_3) / np1_3) * Lx  # x
            _ty = mLx_2 + (math.floor((ix - z * np2_3) / np1_3) / np1_3) * Lx  # y
            _tz = mLx_2 + (z / np1_3) * Lx

            '''Potentially could put particles outside the local domain, reset() should be followed by some BC checking'''
            if (_d[0] <= _tx < _d[1]) and (_d[2] <= _ty < _d[3]) and (_d[4] <= _tz < _d[5]):
                _p[_n, 0] = _tx + random.uniform(-1. * self._dev, self._dev)
                _p[_n, 1] = _ty + random.uniform(-1. * self._dev, self._dev)
                _p[_n, 2] = _tz + random.uniform(-1. * self._dev, self._dev)
                _gid[_n] = ix
                _n += 1

        state_input.set_n(_n)
        _p.halo_start_reset()


################################################################################################################
# PosInitTwoParticlesInABox DEFINITIONS
################################################################################################################   

class PosInitTwoParticlesInABox(object):
    """
    Creates two particles a set distance apart on the  given axis, centred on the origin. Places these within a containing volume of given extents.
    
    :arg double rx: Distance between particles.
    :arg np.array(3,1) extents: Extent for containing volume.
    :arg np.array(3,1) axis: axis to centre on.
    """

    def __init__(self, rx, extent=np.array([1.0, 1.0, 1.0]), axis=np.array([1.0, 0.0, 0.0])):
        self._extent = extent
        self._axis = axis
        self._rx = (0.5 / np.linalg.norm(self._axis)) * rx

    def get_extent(self, state_input):
        """
        Initialise domain extents prior to setting particle positions.
        """
        state_input.domain.set_extent(self._extent)

    def reset(self, state_input):
        """
        Resets the first two particles in the input state domain to sit on the x-axis the set distance apart.
        
        
        :arg state state_input: State object containing at least two particles.
        """
        if state_input.n() >= 2:
            _N = 0
            _d = state_input.domain.boundary

            _tmp = -1. * self._rx * self._axis
            _tmp2 = self._rx * self._axis

            if (_d[0] <= _tmp[0] < _d[1]) and (_d[2] <= _tmp[1] < _d[3]) and (_d[4] <= _tmp[2] < _d[5]):
                state_input.positions[0,] = _tmp
                state_input.global_ids[0] = 0
                _N += 1

            if (_d[0] <= _tmp2[0] < _d[1]) and (_d[2] <= _tmp2[1] < _d[3]) and (_d[4] <= _tmp2[2] < _d[5]):
                state_input.positions[_N,] = _tmp2
                state_input.global_ids[_N] = 1
                _N += 1

            state_input.set_n(_N)


        else:
            print "ERROR: PosInitTwoParticlesInABox, not enough particles!"

            # state_input.domain.set_extent(self._extent)


################################################################################################################
# PosInitOneParticleInABox DEFINITIONS
################################################################################################################   

class PosInitOneParticleInABox(object):
    """
    Creates one particle in a domain of given extents.
    
    :arg double r: particle location.
    :arg np.array(3,1) extents: Extent for containing volume.
    """

    def __init__(self, r=np.array([0.0, 0.0, 0.0]), extent=np.array([1.0, 1.0, 1.0])):
        self._extent = extent
        self._r = r

    def get_extent(self, state_input):
        """
        Initialise domain extents prior to setting particle positions.
        """
        state_input.domain.set_extent(self._extent)

    def reset(self, state_input):
        """
        Resets the first particle in the input state domain to sit on the input point.
        
        :arg state state_input: State object containing at least one particle.
        """

        _N = 0
        _d = state_input.domain.boundary

        if (_d[0] <= self._r[0] < _d[1]) and (_d[2] <= self._r[1] < _d[3]) and (_d[4] <= self._r[2] < _d[5]):
            state_input.positions[0,] = self._r
            _N += 1

        state_input.set_n(_N)
        state_input.global_ids[0] = 0
        state_input.positions.halo_start_reset()
        state_input.velocities.halo_start_reset()


        # state_input.domain.set_extent(self._extent)


################################################################################################################
# PosInitDLPOLYConfig DEFINITIONS
################################################################################################################  

class PosInitDLPOLYConfig(object):
    """
    Read positions from DLPLOY config file.
    
    :arg str filename: Config filename.
    """

    def __init__(self, filename=None):
        self._f = filename
        assert self._f is not None, "No position config file specified"

    def get_extent(self, state_input):
        """
        Initialise domain extents prior to setting particle positions.
        """
        fh = open(self._f)

        extent = np.array([0., 0., 0.])

        for i, line in enumerate(fh):
            if i == 2:
                extent[0] = line.strip().split()[0]
            if i == 3:
                extent[1] = line.strip().split()[1]
            if i == 4:
                extent[2] = line.strip().split()[2]
            else:
                pass

        fh.close()

        assert extent.sum() > 0., "PosInit Error: Bad extent read"

        state_input.domain.set_extent(extent)

    def reset(self, state_input):
        """
        Resets particle positions to those in file.
        
        :arg state state_input: State object containing required number of particles.
        """

        fh = open(self._f)
        shift = 7
        offset = 4
        count = 0
        _n = 0

        _d = state_input.domain.boundary

        for i, line in enumerate(fh):

            if (i > (shift - 2)) and ((i - shift + 1) % offset == 0) and count < state_input.nt():
                _tx = float(line.strip().split()[0])
                _ty = float(line.strip().split()[1])
                _tz = float(line.strip().split()[2])

                if (_d[0] <= _tx < _d[1]) and (_d[2] <= _ty < _d[3]) and (_d[4] <= _tz < _d[5]):

                    state_input.positions[_n, 0] = _tx
                    state_input.positions[_n, 1] = _ty
                    state_input.positions[_n, 2] = _tz

                    state_input.global_ids[_n] = count
                    _n += 1
                else:
                    '''
                    state_input.positions[_n,0]=_tx
                    state_input.positions[_n,1]=_ty
                    state_input.positions[_n,2]=_tz
                    
                    
                    state_input.global_ids[_n]=count
                    _n += 1
                    '''
                    pass

                count += 1

        state_input.set_n(_n)

        fh.close()


############################################################################################################
# VelInitNormDist DEFINITIONS
############################################################################################################


class VelInitNormDist(object):
    """
    Initialise velocities by sampling from a gaussian distribution.
    
    :arg double mu: Mean for gaussian distribution.
    :arg double sig: Standard deviation for gaussian distribution.
    
    """

    def __init__(self, mu=0.0, sig=1.0):
        self._mu = mu
        self._sig = sig

    def reset(self, state_input):
        """
        Resets particle velocities to Gaussian distribution.
        
        :arg state state_input: Input state class oject containing velocities.
        """

        # Get velocities.
        vel_in = state_input.velocities

        # Apply normal distro to velocities.
        for ix in range(state_input.n()):
            vel_in[ix,] = [random.gauss(self._mu, self._sig), random.gauss(self._mu, self._sig),
                           random.gauss(self._mu, self._sig)]


################################################################################################################
# VelInitTwoParticlesInABox DEFINITIONS
################################################################################################################


class VelInitTwoParticlesInABox(object):
    """
    Sets velocities for two particles.
    
    :arg np.array(3,1) vx: Velocity vector for particle 1.
    :arg np.array(3,1) vy: Velocity vector for particle 2.
    
    """

    def __init__(self, vx=np.array([0., 0., 0.]), vy=np.array([0., 0., 0.])):
        self._vx = vx
        self._vy = vy

    def reset(self, state_input):
        """
        Resets the particles in the input state to the required velocities.
        
        :arg state state_input: input state.
        """

        if state_input.nt() >= 2:
            for ix in range(state_input.n()):
                if state_input.global_ids[ix] == 0:
                    state_input.velocities[ix] = self._vx
                elif state_input.global_ids[ix] == 1:
                    state_input.velocities[ix] = self._vy


        else:
            print "ERROR: PosInitTwoParticlesInABox, not enough particles!"


################################################################################################################
# VelInitOneParticleInABox DEFINITIONS
################################################################################################################


class VelInitOneParticleInABox(object):
    """
    Sets velocities for first particle.
    
    :arg np.array(3,1) vx: Velocity vector for particle 1.
    
    """

    def __init__(self, vx=np.array([0., 0., 0.])):
        self._vx = vx

    def reset(self, state_input):
        """
        Resets the particles in the input state to the required velocities.
        
        :arg state state_input: input state.
        """

        if state_input.n() >= 1:
            state_input.velocities[0,] = self._vx


################################################################################################################
# VelInitMaxwellBoltzmannDist DEFINITIONS
################################################################################################################       


class VelInitMaxwellBoltzmannDist(object):
    """
    Initialise velocities by sampling from a gaussian distribution.
    
    :arg double mu: Mean for gaussian distribution.
    :arg double sig: Standard deviation for gaussian distribution.
    
    """

    def __init__(self, temperature=293.15):
        self._t = float(temperature)
        print "Warning not yet functional"

    def reset(self, state_input):
        """
        Resets particle velocities to Maxwell-Boltzmann distribution.
        
        :arg state state_input: Input state class oject containing velocities and masses.
        """

        # Apply MB distro to velocities.
        for ix in range(state_input.n()):
            scale = math.sqrt(self._t / state_input.masses[state_input.types[ix]])
            stmp = scale * math.sqrt(-2.0 * math.log(random.uniform(0, 1)))
            V0 = 2. * math.pi * random.uniform(0, 1)
            state_input.velocities[ix, 0] = stmp * math.cos(V0)
            state_input.velocities[ix, 1] = stmp * math.sin(V0)
            state_input.velocities[ix, 1] = scale * math.sqrt(-2.0 * math.log(random.uniform(0, 1))) * math.cos(
                2. * math.pi * random.uniform(0, 1))


################################################################################################################
# VelInitDLPOLYConfig DEFINITIONS
################################################################################################################


class VelInitDLPOLYConfig(object):
    """
    Read velocities from DLPLOY config file.
    
    :arg str filename: Config filename.
    """

    def __init__(self, filename=None):
        self._f = filename
        assert self._f is not None, "No position config file specified"

    def reset(self, state_input):
        """
        Resets particle velocities to those in file.
        
        :arg state state_input: State object containing required number of particles.
        """

        fh = open(self._f)
        shift = 8
        offset = 4
        count = 0
        _n = 0

        for i, line in enumerate(fh):
            if (i > (shift - 2)) and ((i - shift + 1) % offset == 0) and count < state_input.nt():

                if state_input.global_ids[_n] == count:
                    state_input.velocities[_n, 0] = line.strip().split()[0]
                    state_input.velocities[_n, 1] = line.strip().split()[1]
                    state_input.velocities[_n, 2] = line.strip().split()[2]
                    _n += 1
                count += 1


################################################################################################################
# MassInitTwoAlternating DEFINITIONS
################################################################################################################


class MassInitTwoAlternating(object):
    """
    Class to initialise masses, alternates between two masses.
    
    :arg double m1:  First mass
    :arg double m2:  Second mass
    """

    def __init__(self, m1=1.0, m2=1.0):
        self._m = [m1, m2]

    def reset(self, state):
        """
        Apply to input mass dat class.
        
        :arg Dat mass_input: Dat container with masses.
        """

        '''
        for ix in range(state.n()):
            mass_input[ix] = self._m[(state.global_ids[ix] % 2)]
        '''
        state.masses[0] = self._m[0]
        state.masses[1] = self._m[1]

        for ix in range(state.n()):
            state.types[ix] = state.global_ids[ix] % 2


################################################################################################################
# MassInitIdentical DEFINITIONS
################################################################################################################


class MassInitIdentical(object):
    """
    Class to initialise all masses to one value.
    
    :arg double m: Mass default 1.0
    """

    def __init__(self, m=1.0):
        self._m = float(m)

    def reset(self, state):
        """
        Apply to input mass dat class.
        
        :arg Dat mass_input: Dat container with masses.
        """
        state.masses[0] = self._m

        for ix in range(state.n()):
            state.types[ix] = 0
