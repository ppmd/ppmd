import build
import halo
import host
import loop
import mpi
import pio
import runtime
import state
import numpy as np
import math
import random
import data
import ctypes as ct
import gpucuda
import cell
import domain
import pairloop
import threading
import kernel



class BaseMDSimulation(object):
    """
    Class to setup and contain simulation. Provdies methods to update the kinetic energy and forces of the
    simulation state.


    :arg domain_in: Instance of a domain class to use for the simulation.
    :arg potential_in: Short range potential between particles in simulation.
    :arg particle_pos_init: Method to initialise particle positions with, see PosInit* classes.
    :arg particle_vel_init: Method to initialise particle velocities with, see VelInit* classes.
    :arg particle_mass_init: Method to initialise particle masses with, see MassInit* classes.
    :arg int n: Total number of particles in simulation.
    :arg float cutoff: Cutoff to perform cell decomposition with if a potential is not passed.
    """

    def __init__(self,
                 domain_in,
                 potential_in=None,
                 particle_pos_init=None,
                 particle_vel_init=None,
                 particle_mass_init=None,
                 n=0,
                 cutoff=None,
                 domain_boundary_condition=domain.BoundaryTypePeriodic()):

        self.potential = potential_in
        """Short range potential between particles."""

        # Create a state
        self.state = state.BaseMDState()
        """Simulation state, of type state.* """

        # Add integer attributes to state
        self.state.n = n
        self.state.nt = n

        if potential_in is not None:
            self._cutoff = self.potential.rc
        else:
            self._cutoff = cutoff

        self._cell_width = 1.1 * self._cutoff


        self._boundary_method = domain_boundary_condition
        self._boundary_method.set_state(self.state)

        # Add particle dats
        _factor = 5
        self.state.positions = data.ParticleDat(n, 3, name='positions', max_npart=_factor * n)
        self.state.velocities = data.ParticleDat(n, 3, name='velocities', max_npart=_factor * n)
        self.state.forces = data.ParticleDat(n, 3, name='forces', max_npart=_factor * n)
        self.state.global_ids = data.ParticleDat(n, 1, dtype=ct.c_int, name='global_ids', max_npart=_factor * n)
        self.state.types = data.ParticleDat(n, 1, dtype=ct.c_int, name='types', max_npart=_factor * n)


        # Add typed dats.
        self.state.mass = data.TypedDat(1, n)


        # Add scalar dats.
        # Potential energy.
        self.state.u = data.ScalarArray(ncomp=2, name='potential_energy')
        self.state.u.halo_aware = True

        # Kinetic energy
        self.state.k = data.ScalarArray()

        if gpucuda.INIT_STATUS():
            self.state.positions.add_cuda_dat()
            self.state.forces.add_cuda_dat()
            self.state.u.add_cuda_dat()

        '''
        # gpucuda dats
        if gpucuda.INIT_STATUS():
            gpucuda.CUDA_DATS.register(self.state.u)
            gpucuda.CUDA_DATS.register(self.state.positions)
            gpucuda.CUDA_DATS.register(self.state.forces)
        '''

        # domain TODO: Maybe move domain to simulation not state.
        self.state.domain = domain_in

        # Initialise domain extent
        particle_pos_init.get_extent(self.state)
        # Initialise cell list
        self._cell_structure = cell.cell_list.setup(self.state.as_func('n'), self.state.positions, self.state.domain, self._cell_width)
        # Setup callbacks
        cell.cell_list.setup_update_tracking(self._determine_update_status)
        cell.cell_list.setup_callback_on_update(self._reset_moved_distance)


        if type(self.state.domain) is domain.BaseDomainHalo:
            halo.HALOS = halo.CartesianHalo()

        # add gpu arrays
        if gpucuda.INIT_STATUS():
            cell.cell_list.cell_reverse_lookup.add_cuda_dat()
            cell.cell_list.cell_contents_count.add_cuda_dat()
            cell.cell_list.cell_list.add_cuda_dat()

        # Initialise positions
        particle_pos_init.reset(self.state)

        # Initialise velocities
        if particle_vel_init is not None:
            particle_vel_init.reset(self.state)

        # Init masses
        if particle_mass_init is not None:
            particle_mass_init.reset(self.state)


        # Set state time to 0
        self.state.time = 0.0
        """Time of the state."""

        self._prev_time = 0.0
        self._moved_distance = 0.0

        # short range potential data dict init
        if self.potential is not None:
            _potential_dat_dict = self.potential.datdict(self.state)


        if self._cell_structure and self.potential is not None:
            # Need to pass entire state such that all particle dats can be sorted.
            cell.group_by_cell.setup(self.state)

            # TODO remove these two lines when creating access descriptors.
            cell.cell_list.sort()
            cell.group_by_cell.group_by_cell()
            cell.cell_list.trigger_update()

            # If domain has halos TODO, if when domain gets moved etc
            if type(self.state.domain) is domain.BaseDomainHalo:

                self._forces_update_lib = pairloop.PairLoopRapaportHalo(domain=self.state.domain,
                                                                        potential=self.potential,
                                                                        dat_dict=_potential_dat_dict)
                '''
                self._forces_update_lib2 = pairloop.PairLoopRapaportHaloOpenMP(domain=self.state.domain,
                                                                               potential=self.potential,
                                                                               dat_dict=_potential_dat_dict)


                self._forces_update_lib = pairloop.PairLoopNeighbourListOpenMP(potential=self.potential,
                                                                         dat_dict=_potential_dat_dict)

                self._forces_update_lib2 = pairloop.PairLoopNeighbourList(potential=self.potential,
                                                                         dat_dict=_potential_dat_dict)

                self._forces_update_lib2 = pairloop.PairLoopNeighbourListLayersHybrid(potential=self.potential,
                                                                                     dat_dict=_potential_dat_dict,
                                                                                     openmp=False)
                '''


            # If domain is without halos
            elif type(self.state.domain) is domain.BaseDomain:
                self._forces_update_lib = pairloop.PairLoopRapaport(domain=self.state.domain,
                                                                    potential=self.potential,
                                                                    dat_dict=_potential_dat_dict)

            # create gpu looping method if gpucuda module is initialised
            if gpucuda.INIT_STATUS():
                self.gpu_forces_timer = runtime.Timer(runtime.TIMER, 0)
                if type(self.state.domain) is domain.BaseDomain:
                     self._forces_update_lib_gpucuda = gpucuda.SimpleCudaPairLoop(n=self.state.as_func('n'),
                                                                                  domain=self.state.domain,
                                                                                  potential=self.potential,
                                                                                  dat_dict=_potential_dat_dict)
                if type(self.state.domain) is domain.BaseDomainHalo:
                    self._forces_update_lib_gpucuda = gpucuda.SimpleCudaPairLoopHalo3D(n=self.state.as_func('n'),
                                                                                       domain=self.state.domain,
                                                                                       potential=self.potential,
                                                                                       dat_dict=_potential_dat_dict)

        # If no cell structure was created
        elif self.potential is not None:
            self._forces_update_lib = pairloop.DoubleAllParticleLoopPBC(n=self.state.as_func('n'),
                                                                        domain=self.state.domain,
                                                                        kernel=self.potential.kernel,
                                                                        particle_dat_dict=_potential_dat_dict)

        self.timer = runtime.Timer(runtime.TIMER, 0)
        self.cpu_forces_timer = runtime.Timer(runtime.TIMER, 0)

        if runtime.DEBUG.level > 0:
            pio.pprint("DEBUG IS ON")

        self._kinetic_energy_lib = None

    def _get_max_moved_distance(self):
        """
        Get the maxium distance moved by a particle. First call will always be incorrect
        :return:
        """

        _dt = self.state.time - self._prev_time

        self._prev_time = self.state.time

        if self.state.n > 0:
            return _dt * self.state.velocities.dat[0:self.state.n:].max()
        else:
            return 0.0

    def _determine_update_status(self):
        """
        Return true if update of cell list and neighbour list is needed.
        :return:
        """
        self._moved_distance += self._get_max_moved_distance()

        if self._moved_distance >= 0.5 * (self._cell_width - self._cutoff):
            print "WARNING PARTICLE MOVED TOO FAR, rank:", mpi.MPI_HANDLE.rank

        if (self._moved_distance >= 0.5 * (self._cell_width - self._cutoff)) or \
                (self.state.version_id % 10 == 0) or \
                self.state.invalidate_lists:
            return True
        else:
            # print "False", self._moved_distance, (self._cell_width - self._cutoff)
            return False

    def _reset_moved_distance(self):
        self._moved_distance = 0.0
        self.state.invalidate_lists = False

    def forces_update(self):
        """
        Updates the forces in the simulation state using the short range potential.
        """

        self.timer.start()


        # reset forces
        self.state.forces.set_val(0.)
        self.state.u.scale(0.)

        if gpucuda.INIT_STATUS():
            # copy data to gpu
            t0 = threading.Thread(target=cell.cell_list.cell_list.copy_to_cuda_dat()); t0.start()
            t1 = threading.Thread(target=self.state.positions.copy_to_cuda_dat()); t1.start()
            t2 = threading.Thread(target=cell.cell_list.cell_contents_count.copy_to_cuda_dat()); t2.start()
            t3 = threading.Thread(target=cell.cell_list.cell_reverse_lookup.copy_to_cuda_dat()); t3.start()

        self.cpu_forces_timer.start()
        #if self.state.n > 0:
        #self._forces_update_lib.execute()
        self._forces_update_lib.execute()


        self.cpu_forces_timer.pause()

        if gpucuda.INIT_STATUS():
            t0.join()
            t1.join()
            t2.join()
            t3.join()

            self.gpu_forces_timer.start()
            self._forces_update_lib_gpucuda.execute()
            self.gpu_forces_timer.pause()

        if gpucuda.INIT_STATUS():
            self.state.forces.copy_from_cuda_dat()



        self.timer.pause()

    def kinetic_energy_update(self):
        """
        Update the kinetic energy of the simulation state.
        :return: Kinetic energy of the state.
        """

        if self._kinetic_energy_lib is None:
            _K_kernel_code = '''
            k[0] += (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];
            '''
            _constants_K = []
            _K_kernel = kernel.Kernel('K_kernel', _K_kernel_code, _constants_K)
            self._kinetic_energy_lib = loop.SingleAllParticleLoop(self.state.as_func('n'),
                                                                  self.state.types,
                                                                  _K_kernel,
                                                                  {'V': self.state.velocities, 'k': self.state.k, 'M': self.state.mass})
        self.state.k.dat[0] = 0.0
        self._kinetic_energy_lib.execute()

        return self.state.k.dat

    def execute_boundary_conditions(self):
        """
        Execute the boundary conditions for the simulation.
        """
        self._boundary_method.apply()









########################################################################################################
# Below here is initialisations for positions velocities etc.
########################################################################################################











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

        state_input.n = _n
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



        state_input.n = _n
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
        if state_input.n >= 2:
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

            state_input.n = _N


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

        state_input.n = _N
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

        print extent

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

        print "boundary", _d, "extent", state_input.domain.extent

        for i, line in enumerate(fh):


            if (i > (shift - 2)) and ((i - shift + 1) % offset == 0) and count < state_input.nt:
                _tx = float(line.strip().split()[0])
                _ty = float(line.strip().split()[1])
                _tz = float(line.strip().split()[2])

                if (_d[0] <= _tx < _d[1]) and (_d[2] <= _ty < _d[3]) and (_d[4] <= _tz < _d[5]):

                    state_input.positions[_n, 0] = _tx
                    state_input.positions[_n, 1] = _ty
                    state_input.positions[_n, 2] = _tz

                    print state_input.positions[_n,::]

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

        state_input.n = _n

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
        for ix in range(state_input.n):
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

        if state_input.nt >= 2:
            for ix in range(state_input.n):
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

        if state_input.nt >= 2:
            for ix in range(state_input.n):
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

        if state_input.n >= 1:
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
        for ix in range(state_input.n):
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
            if (i > (shift - 2)) and ((i - shift + 1) % offset == 0) and count < state_input.nt:

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
        for ix in range(state.n):
            mass_input[ix] = self._m[(state.global_ids[ix] % 2)]
        '''

        state.mass[0,0] = self._m[0]
        state.mass[0,1] = self._m[1]

        for ix in range(state.n):
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
        state.mass[0] = self._m

        for ix in range(state.n):
            state.types[ix] = 0




