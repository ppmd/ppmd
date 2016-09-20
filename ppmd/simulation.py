

# system level
from mpi4py import MPI
import numpy as np
import random
import ctypes as ct
import math

# package level
import halo
import loop
import mpi
import pio
import runtime
import state
import data
import cell
import domain
import pairloop
import kernel
import opt



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
                 domain_boundary_condition=None,
                 setup_only=False,
                 pairloop_in = None):

        self.potential = potential_in
        """Short range potential between particles."""

        # Create a state
        self.state = state.BaseMDState()
        """Simulation state, of type state.* """

        # Add integer attributes to state
        self.state.npart_local = n
        self.state.npart = n

        if potential_in is not None:
            self._cutoff = self.potential.rc
        else:
            self._cutoff = cutoff


        # r_n = (1+\delta) * r_c

        self._delta = 0.1

        self._cell_width = (1 + self._delta) * self._cutoff


        # Initilise timers
        self.boundary_method_timer = opt.Timer(runtime.TIMER, 0)
        self.timer = opt.Timer(runtime.TIMER, 0)
        self.cpu_forces_timer = opt.Timer(runtime.TIMER, 0)
        self.kinetic_energy_timer = opt.Timer(runtime.TIMER, 0)


        if domain_boundary_condition is None:
            self._boundary_method = domain.BoundaryTypePeriodic()

        self._boundary_method.set_state(self.state)
        self.state.domain = domain_in

        self.state.domain.boundary_condition = self._boundary_method


        # Add particle dats
        _factor = 1
        self.state.positions = data.PositionDat(n, 3, name='positions')
        self.state.velocities = data.ParticleDat(n, 3, name='velocities')
        self.state.forces = data.ParticleDat(n, 3, name='forces')
        self.state.global_ids = data.ParticleDat(n, 1, dtype=ct.c_int, name='global_ids')
        self.state.types = data.ParticleDat(n, 1, dtype=ct.c_int, name='types')


        # Add typed dats.
        self.state.mass = data.TypedDat(1, n, key=self.state.types)


        # Add scalar dats.
        # Potential energy.
        self.state.u = data.ScalarArray(ncomp=2, name='potential_energy')
        self.state.u.halo_aware = True

        # Kinetic energy
        self.state.k = data.ScalarArray()




        # Initialise domain extent
        particle_pos_init.get_extent(self.state)

        #self._cell_structure = self.state.domain.cell_decompose(self._cell_width)
        self._cell_structure = True



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


        if self._cell_structure and self.potential is not None and not setup_only:


            # If domain has halos TODO, if when domain gets moved etc
            if type(self.state.domain) is domain.BaseDomainHalo:

                if pairloop_in is None:
                    self._forces_update_lib = pairloop.PairLoopNeighbourList(
                        kernel=self.potential.kernel,
                        dat_dict=_potential_dat_dict,
                        shell_cutoff = self._cell_width)
                elif pairloop_in is pairloop.PairLoopRapaportHalo:
                    self._forces_update_lib = pairloop.PairLoopRapaportHalo(
                        kernel=self.potential.kernel,
                        dat_dict=_potential_dat_dict,
                        domain=domain_in)
                else:
                    self._forces_update_lib = pairloop_in(
                        kernel=self.potential.kernel,
                        dat_dict=_potential_dat_dict,
                        shell_cutoff=self._cell_width)


        # If no cell structure was created
        elif self.potential is not None and not setup_only:
            print "Warning check looping method!"
            self._forces_update_lib = pairloop.DoubleAllParticleLoopPBC(
                n=self.state.as_func('npart_local'),
                domain=self.state.domain,
                kernel=self.potential.kernel,
                particle_dat_dict=_potential_dat_dict
            )


        if runtime.DEBUG > 0:
            pio.pprint("DEBUG IS ON")

        self._kinetic_energy_lib = None



    def forces_update(self):
        """
        Updates the forces in the simulation state using the short range potential.
        """

        self.timer.start()

        # reset forces
        #self.state.forces.set_val(0.)

        self.cpu_forces_timer.start()
        self.state.forces.zero(self.state.npart_local)
        self.state.u.zero()
        self.cpu_forces_timer.pause()

        self._forces_update_lib.execute()


        self.timer.pause()




    def kinetic_energy_update(self):
        """
        Update the kinetic energy of the simulation state.
        :return: Kinetic energy of the state.
        """

        if self._kinetic_energy_lib is None:
            _K_kernel_code = '''
            k(0) += (V(0)*V(0) + V(1)*V(1) + V(2)*V(2))*0.5*M(0);
            '''
            _constants_K = []
            _K_kernel = kernel.Kernel('K_kernel', _K_kernel_code, _constants_K)
            self._kinetic_energy_lib = loop.ParticleLoop(self.state.as_func('npart_local'),
                                                         _K_kernel,
                                                         {'V': self.state.velocities, 'k': self.state.k, 'M': self.state.mass})

            self.kinetic_energy_timer = self._kinetic_energy_lib.loop_timer

        self.state.k.data[0] = 0.0
        self._kinetic_energy_lib.execute(self.state.npart_local)


        return self.state.k.data










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

        state_input.npart_local = _n
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



        state_input.npart_local = _n
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
        self._axis = np.array(axis, dtype=ct.c_double)
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


        if state_input.npart_local >= 2:
            _N = 0
            _d = state_input.domain.boundary

            _tmp = -1. * self._rx * self._axis
            _tmp2 = self._rx * self._axis

            if (_d[0] <= _tmp[0] < _d[1]) and (_d[2] <= _tmp[1] < _d[3]) and (_d[4] <= _tmp[2] < _d[5]):
                state_input.positions.data[0,::] = np.array(_tmp[::], dtype=ct.c_double, copy=True)
                state_input.global_ids[0] = 0
                _N += 1

            if (_d[0] <= _tmp2[0] < _d[1]) and (_d[2] <= _tmp2[1] < _d[3]) and (_d[4] <= _tmp2[2] < _d[5]):
                state_input.positions.data[_N,::] = np.array(_tmp2[::], dtype=ct.c_double, copy=True)
                state_input.global_ids[_N] = 1
                _N += 1

            state_input.npart_local = _N


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

        state_input.npart_local = _N
        state_input.global_ids[0] = 0
        state_input.positions.halo_start_reset()
        state_input.velocities.halo_start_reset()


        # state_input.domain.set_extent(self._extent)


################################################################################################################
# PosInitDLPOLYConfig DEFINITIONS
################################################################################################################

def periodic_mod(lower, val, upper):
    extent = upper - lower
    if val < lower:
        val += extent
    elif val > upper:
        val -= extent

    return val


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


            if (i > (shift - 2)) and ((i - shift + 1) % offset == 0) and count < state_input.npart:
                _tx = float(line.strip().split()[0])
                _ty = float(line.strip().split()[1])
                _tz = float(line.strip().split()[2])

                if (_d[0] <= _tx < _d[1]) and (_d[2] <= _ty < _d[3]) and (_d[4] <= _tz < _d[5]):

                    state_input.positions.data[_n, 0] = _tx
                    state_input.positions.data[_n, 1] = _ty
                    state_input.positions.data[_n, 2] = _tz

                    #print state_input.positions.data[_n,::]

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
                    if mpi.MPI_HANDLE.nproc == 1:
                        print "Warning an input position was outside the simulation domain, correcting."

                        print _d[0] , _tx , _d[1],_d[2] , _ty , _d[3],_d[4] , _tz , _d[5]
                        
                        _tx = periodic_mod(_d[0] , _tx , _d[1])
                        _ty = periodic_mod(_d[2] , _ty , _d[3])
                        _tz = periodic_mod(_d[4] , _tz , _d[5])
                            
                        state_input.positions.data[_n, 0] = _tx
                        state_input.positions.data[_n, 1] = _ty
                        state_input.positions.data[_n, 2] = _tz

                        #print state_input.positions.data[_n,::]


                        state_input.global_ids[_n] = count
                        _n += 1

                count += 1

        state_input.npart_local = _n

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
        for ix in range(state_input.npart_local):
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

        if state_input.npart >= 2:
            for ix in range(state_input.npart_local):
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

        if state_input.npart >= 2:
            for ix in range(state_input.npart_local):
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

        if state_input.npart_local >= 1:
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
        for ix in range(state_input.npart_local):
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
            if (i > (shift - 2)) and ((i - shift + 1) % offset == 0) and count < state_input.npart:

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
        for ix in range(state.npart_local):
            mass_input[ix] = self._m[(state.global_ids[ix] % 2)]
        '''

        state.mass[0,0] = self._m[0]
        state.mass[0,1] = self._m[1]

        for ix in range(state.npart_local):
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

        for ix in range(state.npart_local):
            state.types[ix] = 0




