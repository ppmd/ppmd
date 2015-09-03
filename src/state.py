import numpy as np
import particle
import ctypes
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
                self._looping_method_accel = pairloop.PairLoopRapaportHalo(domain=self._domain,
                                                                           potential=self._potential,
                                                                           dat_dict=_potential_dat_dict)
            # If domain is without halos
            elif type(self._domain) is domain.BaseDomain:
                self._looping_method_accel = pairloop.PairLoopRapaport(domain=self._domain,
                                                                       potential=self._potential,
                                                                       dat_dict=_potential_dat_dict)

            if gpucuda.INIT_STATUS():


                self.gpu_forces_timer = runtime.Timer(runtime.TIMER, 0)


                self._accel_comparison = particle.Dat(self._NT, 3, name='accel_compare')
                if type(self._domain) is domain.BaseDomain:
                    self._looping_method_accel_test = gpucuda.SimpleCudaPairLoop(n=self.n,
                                                                                 domain=self._domain,
                                                                                 potential=self._potential,
                                                                                 dat_dict=_potential_dat_dict)
                if type(self._domain) is domain.BaseDomainHalo:
                    self._looping_method_accel_test = gpucuda.SimpleCudaPairLoopHalo2D(n=self.n,
                                                                                       domain=self._domain,
                                                                                       potential=self._potential,
                                                                                       dat_dict=_potential_dat_dict)


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

#######################################################################################################

class AsFunc(object):
    def __init__(self, instance, name):
        self._i = instance
        self._n = name

    def __call__(self):
        return getattr(self._i, self._n)



class BaseMDState(object):

    def __init__(self):
        self.particle_dats = []


    def __setattr__(self, name, value):
        # Add to instance list of particle dats.
        if type(value) is particle.Dat:
            object.__setattr__(self, name, value)
            self.particle_dats.append(name)

        # Any other attribute.
        else:
            object.__setattr__(self, name, value)

    def as_func(self, name):
        """
        Returns a function handle to evaluate the required attribute.
        :param string name: Name of attribute.
        :return: Function handle (of type class: AsFunc)
        """
        return AsFunc(self, name)

