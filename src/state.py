import numpy as np
import particle
import math
import ctypes
import random
import pairloop
import data
import kernel
import loop
import runtime
import pio
import domain
import gpucuda
import cell
import threading


np.set_printoptions(threshold='nan')

################################################################################################################
# BaseMDStatehalo DEFINITIONS
################################################################################################################


class BaseMDStateHalo(object):
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

    def __init__(self, domain_in, potential_in, particle_pos_init=None, particle_vel_init=None, particle_mass_init=None, n=0):
        """
        Initialise class to hold the state of a simulation.
        :arg domain_in domain_in: Container within which the simulation takes place.
        :arg potential_in potential_in: Potential to use between particles.
        :arg int n: Number of particles, default 1.
        :arg double mass: Mass of particles, default 1.0
        """
        self._potential = potential_in
        self._N = n
        self._NT = n
        self._pos = particle.Dat(self._NT, 3, name='positions')
        self._vel = particle.Dat(self._NT, 3, name='velocities')
        self._accel = particle.Dat(self._NT, 3, name='accelerations')

        if gpucuda.INIT_STATUS():
            self._pos.add_cuda_dat()
            self._accel.add_cuda_dat()

        '''Store global ids of particles'''
        self._global_ids = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)

        '''Lookup table between id and particle type'''
        self._types = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)

        '''Mass is an example of a property dependant on particle type'''
        self._mass = particle.TypedDat(self._NT, 1, 1.0)

        self._domain = domain_in

        # potential_in energy, kenetic energy, total energy.
        self._U = data.ScalarArray(max_size=2, name='potential_energy')
        self._U.init_halo_dat()
        if gpucuda.INIT_STATUS():
            self._U.add_cuda_dat()

        self._K = data.ScalarArray()
        self._Q = data.ScalarArray()

        '''Get domain_in extent from position config'''
        particle_pos_init.get_extent(self)

        '''Attempt to initialise cell array'''
        #self._cell_setup_attempt = self._domain.set_cell_array_radius(self._potential._rn)
        cell.cell_list.setup(self.n, self._pos, self._domain, self._potential._rn)

        self._cell_setup_attempt = True

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

        # Setup acceleration updating from given potential_in
        _potential_dat_dict = self._potential.datdict(self)

        if self._cell_setup_attempt is True:

            cell.group_by_cell.setup(self._pos, self._vel, self._global_ids, self._types)
            cell.cell_list.sort()
            cell.group_by_cell.group_by_cell()

            # If domain has halos
            if type(self._domain) is domain.BaseDomainHalo:
                self._looping_method_accel = pairloop.PairLoopRapaportHalo(n=self._N,
                                                                           domain=self._domain,
                                                                           positions=self._pos,
                                                                           potential=self._potential,
                                                                           dat_dict=_potential_dat_dict,
                                                                           cell_list=cell.cell_list.cell_list)
            # If domain is without halos
            elif type(self._domain) is domain.BaseDomain:
                self._looping_method_accel = pairloop.PairLoopRapaport(n=self._N,
                                                                       domain=self._domain,
                                                                       positions=self._pos,
                                                                       potential=self._potential,
                                                                       dat_dict=_potential_dat_dict,
                                                                       cell_list=cell.cell_list.cell_list)

            if gpucuda.INIT_STATUS():


                self.gpu_forces_timer = runtime.Timer(runtime.TIMER, 0)


                self._accel_comparison = particle.Dat(self._NT, 3, name='accel_compare')
                if type(self._domain) is domain.BaseDomain:
                    self._looping_method_accel_test = gpucuda.SimpleCudaPairLoop(n=self.n,
                                                                                 domain=self._domain,
                                                                                 positions=self._pos,
                                                                                 potential=self._potential,
                                                                                 dat_dict=_potential_dat_dict,
                                                                                 cell_list=cell.cell_list.cell_list,
                                                                                 cell_contents_count=cell.cell_list.cell_contents_count,
                                                                                 particle_cell_lookup=cell.cell_list.cell_reverse_lookup)
                if type(self._domain) is domain.BaseDomainHalo:
                    self._looping_method_accel_test = gpucuda.SimpleCudaPairLoopHalo2D(n=self.n,
                                                                                     domain=self._domain,
                                                                                     positions=self._pos,
                                                                                     potential=self._potential,
                                                                                     dat_dict=_potential_dat_dict,
                                                                                     cell_list=cell.cell_list.cell_list,
                                                                                     cell_contents_count=cell.cell_list.cell_contents_count,
                                                                                     particle_cell_lookup=cell.cell_list.cell_reverse_lookup)


        else:
            self._looping_method_accel = pairloop.DoubleAllParticleLoopPBC(n=self._N,
                                                                           domain=self._domain,
                                                                           kernel=self._potential.kernel,
                                                                           particle_dat_dict=_potential_dat_dict)

        self.timer = runtime.Timer(runtime.TIMER, 0)
        self.cpu_forces_timer = runtime.Timer(runtime.TIMER, 0)

        if runtime.DEBUG.level > 0:
            pio.pprint("DEBUG IS ON")



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

    def reset_u(self):
        """
        Reset potential energy to 0.0
        """
        # self._U._Dat = np.zeros([1], dtype=ctypes.c_double, order='C')
        self._U.scale(0.)

    def forces_update(self):
        """
        Updates forces dats using given looping method.
        """

        self.timer.start()

        cell.cell_list.sort()
        cell.group_by_cell.group_by_cell()


        #self._group_by_cell()

        if (self._cell_setup_attempt is True) and (self._domain.halos is not False):
            self._domain.halos.set_position_info(cell.cell_list.cell_contents_count, cell.cell_list.cell_list)
            self._domain.halos.exchange(self._pos)

        self.set_forces(ctypes.c_double(0.0))
        self.reset_u()
        if gpucuda.INIT_STATUS():
            self._U.copy_to_cuda_dat()



        if gpucuda.INIT_STATUS():
            # copy data to gpu
            t0 = threading.Thread(target=cell.cell_list.cell_list.copy_to_cuda_dat())
            t0.start()

            t1 = threading.Thread(target=self._pos.copy_to_cuda_dat())
            t1.start()

            t2 = threading.Thread(target=cell.cell_list.cell_contents_count.copy_to_cuda_dat())
            t2.start()

            t3 = threading.Thread(target=cell.cell_list.cell_reverse_lookup.copy_to_cuda_dat())
            t3.start()




        self.cpu_forces_timer.start()
        if self._N > 0:
            self._looping_method_accel.execute()
            pass

        self.cpu_forces_timer.pause()


        if gpucuda.INIT_STATUS():


            t0.join()
            t1.join()
            t2.join()
            t3.join()


            self.gpu_forces_timer.start()
            self._looping_method_accel_test.execute()
            self.gpu_forces_timer.pause()

            # self._accel.copy_from_cuda_dat()


            # Compare results from gpu with cpu results
            _tol=10**-8

            _u = self._U.dat[0] + 0.5 * self._U.dat[1]
            self._U.copy_from_cuda_dat()
            _ug = (0.5 * self._U.dat[0]) / self._NT

            self._U.dat[0] = _u
            self._U.dat[1] = 0.

            _u /= self._NT

            '''
            if (math.fabs(_u - _ug)) > (_tol * 10):
                print "Potential energy difference:", _u, _ug, math.fabs(_u - _ug)
            '''
            '''
            self._accel.get_cuda_dat().cpy_dth(self._accel_comparison.ctypes_data)

            for ix in range(self._N):


                if (math.fabs(self._accel_comparison.dat[ix,0] - self._accel.dat[ix,0])>_tol) or (math.fabs(self._accel_comparison.dat[ix,1] - self._accel.dat[ix,1])>_tol) or (math.fabs(self._accel_comparison.dat[ix,2] - self._accel.dat[ix,2])>_tol):
                    print "missmatch", ix, self._accel_comparison.dat[ix,:], self._accel.dat[ix,:], self._time, self._particle_cell_lookup[ix], self._domain.cell_array, self._domain.extent
                    # quit()


                pass
            '''



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
    def k(self):
        """
        Return Kenetic energy
        """
        return self._K

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



########################################################################################################
# PosInitLatticeNRho DEFINITIONS
########################################################################################################


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
# VelInitPosBased DEFINITIONS
################################################################################################################


class VelInitPosBased(object):
    """
    Sets velocities for two particles.

    :arg np.array(3,1) vx: Velocity vector for particle 1.
    :arg np.array(3,1) vy: Velocity vector for particle 2.

    """

    def __init__(self):
        pass

    def reset(self, state_input):
        """
        Resets the particles in the input state to the required velocities.

        :arg state state_input: input state.
        """

        if state_input.nt() >= 2:
            for ix in range(state_input.n()):
                if state_input.global_ids[ix] == 0:
                    state_input.velocities[ix,0] = state_input.positions[ix,0] * 10
                    state_input.velocities[ix,1] = state_input.positions[ix,1] * 10
                    state_input.velocities[ix,2] = state_input.positions[ix,2] * 10
                elif state_input.global_ids[ix] == 1:
                    state_input.velocities[ix,0] = state_input.positions[ix,0] * 10
                    state_input.velocities[ix,1] = state_input.positions[ix,1] * 10
                    state_input.velocities[ix,2] = state_input.positions[ix,2] * 10


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
