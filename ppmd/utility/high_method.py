from __future__ import division, print_function#, absolute_import

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import math

from ppmd import method, pairloop, loop


import numpy as np

_GRAPHICS = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _GRAPHICS = False

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    _GRAPHICS = False

# system level
import collections
import ctypes
import os
import re
import datetime
import inspect
import time

# package level
from ppmd import kernel, data, runtime, pio, mpi, opt, access

_MPI = mpi.MPI
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier


class IntegratorVelocityVerlet(object):
    def __init__(
            self,
            positions,
            forces,
            velocities,
            masses,
            force_updater,
            interaction_cutoff,
            list_reuse_count,
            looping_method=None
        ):

        self._p = positions
        self._f = forces
        self._v = velocities
        self._m = masses
        self._f_updater = force_updater

        self._delta = float(self._f_updater.shell_cutoff) - \
                      float(interaction_cutoff)

        self._g = positions.group

        self._update_controller = method.ListUpdateController(
            self._g,
            step_count=list_reuse_count,
            velocity_dat=self._v,
            timestep=None,
            shell_thickness=self._delta
        )

        self._p1 = None
        self._p2 = None

        if looping_method is None:
            self._looping_method = loop.ParticleLoop
        else:
            self._looping_method = looping_method

        _suc = self._update_controller
        self._g.get_cell_to_particle_map().setup_pre_update(
            _suc.pre_update
        )

        self._g.get_cell_to_particle_map().setup_update_tracking(
            _suc.determine_update_status
        )

        self._g.get_cell_to_particle_map().setup_callback_on_update(
            _suc.post_update
        )



    def _build_libs(self, dt):
        kernel1_code = '''
        const double M_tmp = 1.0/M(0);
        V(0) += dht*F(0)*M_tmp;
        V(1) += dht*F(1)*M_tmp;
        V(2) += dht*F(2)*M_tmp;
        P(0) += dt*V(0);
        P(1) += dt*V(1);
        P(2) += dt*V(2);
        '''

        kernel2_code = '''
        const double M_tmp = 1.0/M(0);
        V(0) += dht*F(0)*M_tmp;
        V(1) += dht*F(1)*M_tmp;
        V(2) += dht*F(2)*M_tmp;
        '''
        constants = [
            kernel.Constant('dt', dt),
            kernel.Constant('dht',0.5*dt),
        ]

        kernel1 = kernel.Kernel('vv1', kernel1_code, constants)
        self._p1 = self._looping_method(
            kernel=kernel1,
            dat_dict={'P': self._p(access.W),
                               'V': self._v(access.W),
                               'F': self._f(access.R),
                               'M': self._m(access.R)}
        )

        kernel2 = kernel.Kernel('vv2', kernel2_code, constants)
        self._p2 = self._looping_method(
            kernel=kernel2,
            dat_dict={'V': self._v(access.W),
                               'F': self._f(access.R),
                               'M': self._m(access.R)}
        )


    def integrate(self, dt, t, schedule=None):

        self._update_controller.set_timestep(dt)
        self._build_libs(dt)


        #self._f.zero(self._g.npart_local)
        self._f_updater.execute()

        #print self._g.u[0]

        for i in range( int( math.ceil( float(t) / float(dt) ) ) ):

            #print self._g.u[0]

            self._p1.execute(self._g.npart_local)
            #self._f.zero()
            self._f_updater.execute()
            self._p2.execute(self._g.npart_local)

            self._update_controller.increment_step_count()

            if schedule is not None:
                schedule.tick()

            #print 60*'-'
            #print np.max(self._f[0:self._g.npart_local:,:]), np.min(self._f[0:self._g.npart_local:,:])
            #print np.max(self._v[0:self._g.npart_local:,:]), np.min(self._v[0:self._g.npart_local:,:])



###############################################################################
# Velocity Verlet Method
###############################################################################

class VelocityVerlet(object):
    """
    Class to apply Velocity-Verlet to a given state using a given looping
    method.
    
    :arg double dt: Time step size, can be specified at integrate call.
    :arg double T: End time, can be specified at integrate call.
    :arg bool USE_C: Flag to use C looping and kernel.
    :arg DrawParticles plot_handle: PLotting class to plot state at certain
    progress points.
    :arg BasicEnergyStore energy_handle: Energy storage class to log energy at
    each iteration.
    :arg bool writexyz: Flag to indicate writing of xyz at each DT.
    :arg bool DEBUG: Flag to enable debug flags.
    """
    
    def __init__(self, dt=0.0001, t=0.01, simulation=None, schedule=None, shell_thickness=0.0):
    
        self._dt = dt
        self._T = t

        self._sim = simulation
        self._state = self._sim.state

        self._delta = shell_thickness


        self.timer = opt.SynchronizedTimer(runtime.TIMER)
        
        self._domain = self._state.domain
        self._N = self._state.npart_local
        self._A = self._state.forces
        self._V = self._state.velocities
        self._P = self._state.positions
        self._M = self._state.mass
        self._K = self._state.k

        self._schedule = schedule
        
        self._kernel1_code = '''
        const double M_tmp = 1.0/M(0);
        V(0) += dht*A(0)*M_tmp;
        V(1) += dht*A(1)*M_tmp;
        V(2) += dht*A(2)*M_tmp;
        P(0) += dt*V(0);
        P(1) += dt*V(1);
        P(2) += dt*V(2);
        '''
                
        self._kernel2_code = '''
        const double M_tmp = 1.0/M(0);
        V(0) += dht*A(0)*M_tmp;
        V(1) += dht*A(1)*M_tmp;
        V(2) += dht*A(2)*M_tmp;
        '''

        self._p1 = None
        self._p2 = None


        #### NEW

        self._update_controller = method.ListUpdateController(
            self._state,
            step_count=10,
            velocity_dat=self._state.velocities,
            timestep=self._dt,
            shell_thickness=self._delta
        )

        _suc = self._update_controller

        self._state.get_cell_to_particle_map().setup_pre_update(_suc.pre_update)
        self._state.get_cell_to_particle_map().setup_update_tracking(_suc.determine_update_status)
        self._state.get_cell_to_particle_map().setup_callback_on_update(_suc.post_update)


    @property
    def timer1(self):
        if self._p1 is not None:
            return self._p1.loop_timer
        else:
            return None

    @property
    def timer2(self):
        if self._p2 is not None:
            return self._p2.loop_timer
        else:
            return None

    def integrate(self, dt = None, t = None):
        """
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double t: End time.
        """
        print("starting integration")
        if dt is not None:
            self._dt = dt
        if t is not None:
            self._T = t

        self._max_it = int(math.ceil(self._T/self._dt))

        self._constants = [kernel.Constant('dt',self._dt), kernel.Constant('dht',0.5*self._dt),]

        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
        self._p1 = loop.ParticleLoop(
            self._kernel1,
            {'P':self._P(access.W),
            'V':self._V(access.W),
            'A':self._A(access.R), 
            'M':self._M(access.R)}
        )

        self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
        self._p2 = loop.ParticleLoop(
            self._kernel2,
            {'V':self._V(access.W),
            'A':self._A(access.R), 
            'M':self._M(access.R)}
        )


        self._update_controller.execute_boundary_conditions()

        self._sim.forces_update()

        self.timer.start()
        self._velocity_verlet_integration()
        self.timer.pause()




    def _velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """

        #self._sim.execute_boundary_conditions()

        #self._sim.forces_update()

        for i in range(self._max_it):
            # print _MPIRANK, "-------", i , self._state.npart_local

            self._p1.execute(self._state.npart_local)


            self._sim.forces_update()
            self._p2.execute(self._state.npart_local)

            self._sim.kinetic_energy_update()

            #self._state.time += self._dt
            self._update_controller.increment_step_count()


            if self._schedule is not None:
                self._schedule.tick()



###############################################################################
# Anderson thermostat
###############################################################################


class VelocityVerletAnderson(VelocityVerlet):
    
    def integrate_thermostat(self, dt=None, t=None, temp=273.15, nu=1.0):
        """
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double t: End time.
        :arg double temp: Temperature of heat bath.
        """
        
        self._Temp = temp
        self._nu = nu
        
        if dt is not None:
            self._dt = dt
        if t is not None:
            self._T = t
            
        self._max_it = int(math.ceil(self._T/self._dt))

        self._constants1 = [kernel.Constant('dt',self._dt), kernel.Constant('dht',0.5*self._dt),]
        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants1)
        self._p1 = loop.ParticleLoop(self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M})

        self._kernel2_thermostat_code = '''

        //Anderson thermostat here.
        //probably horrific random code.

        const double tmp_rand_max = 1.0/RAND_MAX;

        if (rand()*tmp_rand_max < rate) {

            //Box-Muller method.


            const double scale = sqrt(temperature/M(0));
            const double stmp = scale*sqrt(-2.0*log(rand()*tmp_rand_max));

            const double V0 = 2.0*M_PI*rand()*tmp_rand_max;
            V(0) = stmp*cos(V0);
            V(1) = stmp*sin(V0);
            V(2) = scale*sqrt(-2.0*log(rand()*tmp_rand_max))*cos(2.0*M_PI*rand()*tmp_rand_max);

        }
        else {
            const double M_tmp = 1./M(0);
            V(0) += dht*A(0)*M_tmp;
            V(1) += dht*A(1)*M_tmp;
            V(2) += dht*A(2)*M_tmp;
        }

        '''

        self._constants2_thermostat = [kernel.Constant('rate',self._dt*self._nu), kernel.Constant('dt',self._dt), kernel.Constant('dht',0.5*self._dt), kernel.Constant('temperature',self._Temp),]

        self._kernel2_thermostat = kernel.Kernel('vv2_thermostat',self._kernel2_thermostat_code,self._constants2_thermostat, headers = ['math.h','stdlib.h','time.h','stdio.h'])
        self._p2_thermostat = loop.ParticleLoop(self._kernel2_thermostat,{'V':self._V,'A':self._A, 'M':self._M})

        _t = ppmd.opt.Timer(runtime.TIMER, 0, start=True)
        self._velocity_verlet_integration_thermostat()
        _t.stop("VelocityVerletAnderson")
    
    def _velocity_verlet_integration_thermostat(self):
        """
        Perform Velocity Verlet integration up to time T.
        """

        self._domain.bc_execute()
        self._state.forces_update()

        for i in range(self._max_it):
              
            self._p1.execute()
            
            # update forces
            self._domain.bc_execute()
            self._state.forces_update()
            
            self._p2_thermostat.execute()

            self._state.kinetic_energy_update()
            self._state.add_time(self._dt)

            if self._schedule is not None:
                self._schedule.tick()




###############################################################################
# KE
###############################################################################

class KineticEnergyTracker(object):
    def __init__(
            self,
            velocities=None,
            masses=None,
            kinetic_energy_dat=None,
            looping_method=None
        ):

        if looping_method is None:
            looping_method = loop.ParticleLoop
        if kinetic_energy_dat is None:
            self.k = data.ScalarArray(ncomp=1, dtype=ctypes.c_double)
        else:
            self.k = kinetic_energy_dat

        self._v = velocities

        if looping_method is None:
            looping_method = loop.ParticleLoop

        _K_kernel_code = '''
        k(0) += (V(0)*V(0) + V(1)*V(1) + V(2)*V(2))*0.5*M(0);
        '''
        _constants_K = []
        _K_kernel = kernel.Kernel('K_kernel', _K_kernel_code, _constants_K)
        self._kinetic_energy_lib = looping_method(
            kernel=_K_kernel,
            dat_dict={'V': velocities(access.R),
                               'k': self.k(access.INC),
                               'M': masses(access.R)}
        )

        self._ke_store = []

    def execute(self):
        self.k[0] = 0.0
        self._kinetic_energy_lib.execute(n=self._v.group.npart_local)
        self._ke_store.append(self.k[0])

    def get_kinetic_energy_array(self):
        arr = np.array(self._ke_store, dtype=ctypes.c_double)
        rarr = np.zeros_like(arr)
        _MPI.COMM_WORLD.Allreduce(
            arr,
            rarr
        )
        return rarr

###############################################################################
# PE
###############################################################################

class PotentialEnergyTracker(object):
    def __init__(
            self,
            potential_energy_dat
        ):

        self._u = potential_energy_dat
        self._u_store = []

    def execute(self):
        self._u_store.append(self._u[0])

    def get_potential_energy_array(self):
        arr = np.array(self._u_store, dtype=ctypes.c_double)
        rarr = np.zeros_like(arr)
        _MPI.COMM_WORLD.Allreduce(
            arr,
            rarr
        )
        return rarr



###############################################################################
# G(R)
###############################################################################


class RadialDistributionPeriodicNVE(object):
    """
    Class to calculate radial distribution function.
    
    :arg state state: State containing particle positions.
    :arg double rmax: Maximum radial distance.
    :arg int rsteps: Resolution to record to, default 100.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, state, rmax=None, rsteps=100):
        
        self._count = 0
        self._state = state
        self._extent = self._state.domain.extent
        self._P = self._state.positions
        self._N = self._state.npart_local
        self._rmax = rmax
        
        if self._rmax is None:
            self._rmax = 0.5*np.min(self._extent.data)
        
        
        self._rsteps = rsteps
        
        self._gr = data.ScalarArray(ncomp=self._rsteps, dtype=ctypes.c_int)
        self._gr.scale(0.0)
        
        _headers = ['math.h', 'stdio.h']
        _kernel = '''
        
        
        double R0 = P(1, 0) - P(0, 0);
        double R1 = P(1, 1) - P(0, 1);
        double R2 = P(1, 2) - P(0, 2);
        
        if (abs_md(R0) > exto20 ) { R0 += isign(R0) * extent0 ; }
        if (abs_md(R1) > exto21 ) { R1 += isign(R1) * extent1 ; }
        if (abs_md(R2) > exto22 ) { R2 += isign(R2) * extent2 ; }
        
        const double r2 = R0*R0 + R1*R1 + R2*R2;
        
        if (r2 < rmax2){
            
            double r20=0.0, r21 = r2;
            
            r21 = sqrt(r2);
            #pragma omp atomic
            GR[(int) (abs_md(r21* rstepsoverrmax))]++;
            
        }
        '''
        
        _constants=(kernel.Constant('rmaxoverrsteps', 0.2*self._rmax/self._rsteps ),
                    kernel.Constant('rstepsoverrmax', self._rsteps/self._rmax ),
                    kernel.Constant('rmax2', self._rmax**2 ),
                    kernel.Constant('extent0', self._extent[0] ),
                    kernel.Constant('extent1', self._extent[1] ),
                    kernel.Constant('extent2', self._extent[2] ),
                    kernel.Constant('exto20', 0.5*self._extent[0] ),
                    kernel.Constant('exto21', 0.5*self._extent[1] ),
                    kernel.Constant('exto22', 0.5*self._extent[2] )
                    )

        _grkernel = kernel.Kernel('radial_distro_periodic_static', _kernel, _constants, headers=_headers)
        _datdict = {'P':self._P, 'GR':self._gr}

        self._p = pairloop.DoubleAllParticleLoop(self._N, kernel=_grkernel, dat_dict=_datdict)

        self.timer = ppmd.opt.Timer(runtime.TIMER, 0)

    def evaluate(self):
        """
        Evaluate the radial distribution function.
        """
        
        assert self._rmax <= 0.5*self._state.domain.extent.min, "Maximum radius too large."

        self.timer.start()

        self._p.execute()
        self._count += 1

        self.timer.pause()
        
    def _scale(self):
        self._r = np.linspace(0.+0.5*(self._rmax/self._rsteps), self._rmax-0.5*(self._rmax/self._rsteps), num=self._rsteps, endpoint=True)
        self._grscaled = self._gr.data*self._state.domain.volume/((self._N())*(self._N() - 1)*2*math.pi*(self._r**2) * (self._rmax/float(self._rsteps))*self._count)
        
    def plot(self):

        if _GRAPHICS:

            self._scale()
            if self._count > 0:
                plt.ion()
                _fig = plt.figure()
                _ax = _fig.add_subplot(111)

                plt.plot(self._r, self._grscaled)
                _ax.set_title('Radial Distribution Function')
                _ax.set_xlabel('r')
                _ax.set_ylabel('G(r)')
                plt.show()
            else:
                print("Warning: run evaluate() at least once before plotting.")

    def reset(self):
        self._gr.scale(0.0)

    def raw_write(self, dir_name='./output', filename=None, rename_override=False):
        """
        Function to write Radial Distribution Evaluations to disk.
        
        :arg str dir_name: directory to write to, default ./output.
        :arg str filename: Filename to write to, default array name or data.rdf if name unset.
        :arg bool rename_override: Flagging as True will disable autorenaming of output file.
        """

        if filename is None:
            filename = 'data.rdf'
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if os.path.exists(os.path.join(dir_name, filename)) & (rename_override is not True):
            filename = re.sub('.rdf', datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.rdf', filename)
            if os.path.exists(os.path.join(dir_name,filename)):
                filename = re.sub('.rdf', datetime.datetime.now().strftime("_%f") + '.rdf', filename)
                assert os.path.exists(os.path.join(dir_name,filename)), "raw_write Error: No unquie name found."
        
        self._scale()
        f=open(os.path.join(dir_name,filename),'w')
        
        
        f.write('r \t g(r)\n')
        for ix in range(self._gr.ncomp):
            f.write(str(self._r[ix]) + '\t' + str(self._grscaled[ix]) + '\n')

        f.close()


###############################################################################
# WriteTrajectoryXYZ
###############################################################################


class WriteTrajectoryXYZ(object):
    """
    Write Positions to file in XYZ format from given state to given filename.
    """
    def __init__(self, state=None, dir_name='./output', file_name='out.xyz', title='A', symbol='A' ,overwrite=True, ordered=False):

        assert state is not None, "Error: no state passed"

        self._s = state
        self._title = title
        self._symbol = symbol
        self._ordered = ordered

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if os.path.exists(os.path.join(dir_name, file_name)) & (overwrite is not True):
            file_name = re.sub('.xyz', datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.xyz', file_name)
            if os.path.exists(os.path.join(dir_name, file_name)):
                file_name = re.sub('.xyz', datetime.datetime.now().strftime("_%f") + '.xyz', file_name)
                assert os.path.exists(os.path.join(dir_name, file_name)), "WriteTrajectoryXYZ Error:" \
                                                                          " No unique name found."

        self._fn = file_name
        self._dn = dir_name
        self._fh = None

        if _MPIRANK == 0:
            self._fh = open(os.path.join(self._dn, self._fn), 'w')
            self._fh.close()

        self.timer = ppmd.opt.Timer(runtime.TIMER, 0)

    def write(self):
        """
        Append current positions to file.
        :return:
        """

        self.timer.start()

        space = ' '

        if _MPIRANK == 0:
            self._fh = open(os.path.join(self._dn, self._fn), 'a')
            self._fh.write(str(self._s.npart) + '\n')
            self._fh.write(str(self._title) + '\n')
            self._fh.flush()
        _MPIBARRIER()

        if self._ordered is False:
            for iz in range(_MPISIZE):
                if _MPIRANK == iz:
                    self._fh = open(os.path.join(self._dn, self._fn), 'a')
                    for ix in range(self._s.npart):
                        self._fh.write(str(self._symbol).rjust(3))
                        for iy in range(3):
                            self._fh.write(space + str('%.5f' % self._s.positions[ix, iy]))
                        self._fh.write('\n')

                    self._fh.flush()
                    self._fh.close()

                _MPIBARRIER()

        self.timer.pause()

###############################################################################
# Schedule
###############################################################################


class Schedule(object):
    """
    Class to schedule automated running of functions every set number of steps.

    :arg list steps: List of steps between each run.
    :arg list items: List of functions to run after set number of steps.
    """
    
    def __init__(self, steps=None, items=None):
        self._s = collections.defaultdict(list)

        if (steps is not None) and (items is not None):
            assert len(steps) == len(items), "Schedule error, mis-match between number of steps and number of items."
            for ix in zip(steps, items):
                if (ix[0] > 0) and (ix[1] is not None):
                    assert (inspect.isfunction(ix[1]) or inspect.ismethod(ix[1])) is True, "Schedule error: Passed argument" \
                                                                                           " is not a function/method."

                    self._s[ix[0]].append(ix[1])
                else:
                    pio.pprint("Schedule warning: steps<1 and None type functions will be ignored.")

        self._count = 0

    @property
    def schedule(self):
        """
        Returns the held schedule.
        """
        return self._s
        
    def add_item(self, steps=None, items=None):
        """
        Add an item to the schedule.
        
        :arg function item: Function to run.
        :arg int step: Number of steps between running the function. 
        """

        assert (steps is not None) and (items is not None), "Schedule add_item error: both arguments must be passed"
        assert len(steps) == len(items), "Schedule item_add error, mis-match between" \
                                         " number of steps and number of items."
        for ix in zip(steps, items):
            if ix[0] < 1:
                print("Schedule warning: 0 step items will be ignored.")
            else:
                self._s[ix[0]].append(ix[1])

    @property
    def count(self):
        """
        Return the currently held count
        """
        return self._count
        
    def tick(self):
        """
        Method ran by integrator or other method per iteration. If the required number of ticks have passed the
        required functions will be ran.
        """
        self._count += 1

        for ix in self._s.keys():
            if self._count % ix == 0:
                for iy in self._s[ix]:
                    iy()


################################################################################################################
# VAF
################################################################################################################


class VelocityAutoCorrelation(object):
    """
    Method to calculate Velocity Autocorrelation Function.

    :arg state state: Input state containing velocities.
    :arg int size: Initial length of VAF array (optional).
    :arg data.ParticleDat V0: Initial velocity Dat (optional).
    """

    def __init__(self, state, size=0, v0=None):
        self._state = state
        self._V0 = data.ParticleDat(self._state.npart_local, 3, name='v0')
        self._VT = state.velocities

        self._VO_SET = False
        if v0 is not None:
            self.set_v0(v0)
        else:
            self.set_v0(state=self._state)

        self._VAF = data.ScalarArray(ncomp=1)
        self._V = []
        self._T = []

        _headers = ['stdio.h']
        _constants = None
        _kernel_code = '''

        VAF(0) += (v0(0)*VT(0) + v0(1)*VT(1) + v0(2)*VT(2))*Ni;

        '''
        _reduction = (kernel.Reduction('VAF', 'VAF[I]', '+'),)

        _static_args = {'Ni': ctypes.c_double}

        _kernel = kernel.Kernel('VelocityAutocorrelation', _kernel_code, _constants, _headers, _reduction, _static_args)

        self._datdict = {'VAF': self._VAF, 'v0': self._V0, 'VT': self._VT}

        self._loop = loop.ParticleLoop(self._state.as_func('npart_local'), None, kernel=_kernel, dat_dict=self._datdict)

    def set_v0(self, v0=None, state=None):
        """
        Set an initial velocity Dat to use as V_0. Requires either a velocity Dat or a state as an argument. V_0 will be set to either the passed velocities or to the velocities in the passed state.

        :arg data.ParticleDat v0: Velocity Dat.
        :arg state state: State class containing velocities.
        """

        if v0 is not None:
            self._V0.data = np.copy(v0.data)
            self._V0_SET = True
        if state is not None:
            self._V0.data = np.copy(state.velocities.data)
            self._V0_SET = True
        assert self._V0_SET is True, "No velocities set, check input data."


    def evaluate(self):
        """
        Evaluate VAF using the current velocities held in the state with the velocities in V0.

        :arg double t: Time within block of integration.
        """
        if runtime.TIMER > 0:
            start = time.time()

        _t = self._state.time

        _Ni = 1./self._state.as_func('npart_local')

        self._datdict['VT'] = self._state.velocities
        self._loop.execute(None, self._datdict, {'Ni': ctypes.c_double(_Ni)})

        self._V.append(self._VAF[0])
        self._T.append(_t)

        if runtime.TIMER > 0:
            end = time.time()
            pio.pprint("VAF time taken:", end - start, "s")

    def plot(self):
        """
        Plot array of recorded VAF evaluations.
        """

        if _GRAPHICS:

            _Vloc = np.array(self._V)
            _V = np.zeros(len(self._T))

            print(_Vloc)

            _MPI.COMM_WORLD.Reduce(_Vloc, _V, _MPI.SUM, 0)

            if len(self._T) > 0:
                plt.ion()
                _fig = plt.figure()
                _ax = _fig.add_subplot(111)

                plt.plot(self._T, _V)
                _ax.set_title('Velocity Autocorrelation Function')
                _ax.set_xlabel('Time')
                _ax.set_ylabel('VAF')
                plt.show()
            else:
                print("Warning: run evaluate() at least once before plotting.")


################################################################################################################
# DrawParticles
################################################################################################################


class DrawParticles(object):
    """
    Class to plot n particles with given positions.

    :arg int n: Number of particles.
    :arg np.array(n,3) pos: particle positions.
    :arg np.array(3,1) extent:  domain extents.

    """

    def __init__(self, state=None):

        assert state is not None, "DrawParticles error: no state passed."

        self._state = state

        self._Dat = None
        self._gids = None
        self._pos = None
        self._gid = None

        self._N = None
        self._NT = None
        self._extents = None

        if (_MPIRANK == 0) and _GRAPHICS:
            plt.ion()
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._key = ['red', 'blue']
            plt.show(block=False)

    def norm_vec(self, in_vec):
        norm=np.linalg.norm(in_vec)

        if norm < 10**(-13) :
           return in_vec
        return in_vec/norm

    def draw(self):
        """
        Update current plot, use for real time plotting.
        """

        if _GRAPHICS:

            self._N = self._state.npart_local
            self._NT = self._state.npart
            self._extents = self._state.domain.extent

            '''Case where all particles are local'''
            if _MPISIZE == 1:
                self._pos = self._state.positions
                self._gid = self._state.global_ids

            else:
                '''Need an mpi handle if not all particles are local'''

                '''Allocate if needed'''
                if self._Dat is None:
                    self._Dat = data.ParticleDat(self._NT, 3)
                else:
                    self._Dat.resize(self._NT)

                if self._gids is None:
                    self._gids = data.ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)
                else:
                    self._gids.resize(self._NT)

                _MS = mpi.Status()

                if _MPIRANK == 0:

                    '''Copy the local data.'''
                    self._Dat.data[0:self._N:, ::] = self._state.positions.data[0:self._N:, ::]
                    self._gids.data[0:self._N:] = self._state.global_ids.data[0:self._N:, 0]

                    _i = self._N  # starting point pos
                    _ig = self._N  # starting point gids

                    for ix in range(1, _MPISIZE):
                        _MPIWORLD.Recv(self._Dat.data[_i::, ::], ix, ix, _MS)
                        _i += _MS.Get_count(mpi.mpi_map[self._Dat.dtype]) // 3

                        _MPIWORLD.Recv(self._gids.data[_ig::], ix, ix, _MS)
                        _ig += _MS.Get_count(mpi.mpi_map[self._gids.dtype])

                    self._pos = self._Dat
                    self._gid = self._gids
                else:

                    _MPIWORLD.Send(self._state.positions.data[0:self._N:, ::], 0, _MPIRANK)
                    _MPIWORLD.Send(self._state.global_ids.data[0:self._N:], 0, _MPIRANK)

            if _MPIRANK == 0:


                plt.cla()
                plt.ion()
                for ix in range(self._pos.npart_local):
                    self._ax.scatter(self._pos.data[ix, 0], self._pos.data[ix, 1], self._pos.data[ix, 2],
                                     color=self._key[self._gid[ix] % 2])

                    if _MPISIZE == 1:

                        self._ax.plot((self._pos.data[ix, 0], self._pos.data[ix, 0] + self.norm_vec(self._state.forces.data[ix, 0])),
                                      (self._pos.data[ix, 1], self._pos.data[ix, 1] + self.norm_vec(self._state.forces.data[ix, 1])),
                                      (self._pos.data[ix, 2],self._pos.data[ix, 2] + self.norm_vec(self._state.forces.data[ix, 2])),
                                      color=self._key[self._gid[ix] % 2],
                                      linewidth=2)




                self._ax.set_xlim([-0.5 * self._extents[0], 0.5 * self._extents[0]])
                self._ax.set_ylim([-0.5 * self._extents[1], 0.5 * self._extents[1]])
                self._ax.set_zlim([-0.5 * self._extents[2], 0.5 * self._extents[2]])





                self._ax.set_xlabel('x')
                self._ax.set_ylabel('y')
                self._ax.set_zlabel('z')

                plt.draw()
                plt.show(block=False)

    def cleanup(self):
        plt.close("all")

####################################################################################################
# Energy Store
####################################################################################################


class EnergyStore(object):
    """
    Class to hold energy data more sensibly

    :arg state state: Input state to track energy of.
    """

    def __init__(self, state=None):

        assert state is not None, "EnergyStore error, no state passed."

        self._state = state

        self._t = []
        self._k = []
        self._u = []
        self._q = []


    def update(self):
        """
        Update energy tracking of tracked state.
        :return:
        """

        _k = 0.0
        _u = 0.0
        _q = 0.0
        _t = self._state.time

        if self._state.npart_local > 0:
            # print self._state.u[0], self._state.u[1]

            _U_tmp = self._state.u.data[0]/self._state.npart
            _U_tmp += 0.5*self._state.u.data[1]/self._state.npart
            _u = _U_tmp

            _k = self._state.k[0]/self._state.npart
            _q = _U_tmp+(self._state.k[0])/self._state.npart


        self._k.append(_k)
        self._u.append(_u)
        self._q.append(_q)
        self._t.append(_t)



    def plot(self, _plot = True):
        """
        Plot the stored energy data.

        :return:
        """

        assert len(self._t) > 0, "EnergyStore error, no data to plot"

        self._T_store = data.ScalarArray(self._t)
        self._K_store = data.ScalarArray(self._k)
        self._U_store = data.ScalarArray(self._u)
        self._Q_store = data.ScalarArray(self._q)


        '''REPLACE THIS WITH AN MPI4PY REDUCE CALL'''

        if _MPISIZE > 1:

            # data to collect
            _d = [self._Q_store.data, self._U_store.data, self._K_store.data]

            # make a temporary buffer.
            if _MPIRANK == 0:
                _buff = data.ScalarArray(ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _T = self._T_store.data
                _Q = data.ScalarArray( ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _U = data.ScalarArray(ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _K = data.ScalarArray(ncomp=self._T_store.ncomp, dtype=ctypes.c_double)

                _Q.data[::] += self._Q_store.data[::]
                _U.data[::] += self._U_store.data[::]
                _K.data[::] += self._K_store.data[::]

                _dl = [_Q.data, _U.data, _K.data]
            else:
                _dl = [None, None, None]

            for _di, _dj in zip(_d, _dl):

                if _MPIRANK == 0:
                    _MS = mpi.Status()
                    for ix in range(1, _MPISIZE):
                        _MPIWORLD.Recv(_buff.data[::], ix, ix, _MS)
                        _dj[::] += _buff.data[::]

                else:
                    _MPIWORLD.Send(_di[::], 0, _MPIRANK)

            if _MPIRANK == 0:
                _Q = _Q.data
                _U = _U.data
                _K = _K.data

        else:
            _T = self._T_store.data
            _Q = self._Q_store.data
            _U = self._U_store.data
            _K = self._K_store.data



        if (_MPIRANK == 0) and _GRAPHICS:
            print("last total", _Q[-1])
            print("last kinetic", _K[-1])
            print("last potential", _U[-1])
            print("=============================================")
            print("first total", _Q[0])
            print("first kinetic", _K[0])
            print("first potential", _U[0])

            if _plot:

                plt.ion()
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)

                ax2.plot(_T, _Q, color='r', linewidth=2)
                ax2.plot(_T, _U, color='g')
                ax2.plot(_T, _K, color='b')

                ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kinetic energy')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Energy')

                fig2.canvas.draw()
                plt.show(block=False)

        if _MPIRANK == 0:
            if not os.path.exists(os.path.join(os.getcwd(),'./output')):
                os.system('mkdir ./output')


            _fh = open('./output/energy.txt', 'w')
            _fh.write("Time Kinetic Potential Total\n")
            for ix in range(len(self._t)):
                _fh.write("%(T)s %(K)s %(P)s %(Q)s\n" % {'T':_T[ix], 'K':_K[ix], 'P':_U[ix], 'Q':_Q[ix]})
            _fh.close()

        if (_MPIRANK == 0) and not _GRAPHICS:
            print("last total", _Q[-1])
            print("last kinetic", _K[-1])
            print("last potential", _U[-1])
            print("=============================================")
            print("first total", _Q[0])
            print("first kinetic", _K[0])
            print("first potential", _U[0])

####################################################################################################
# Percentage Printer
####################################################################################################

class PercentagePrinter(object):
    """
    Class to print percentage completion to console.

    :arg float dt: Time step size.
    :arg float t: End time.
    :arg int percent: Percent to print on.
    """
    def __init__(self, dt, t, percent):
        _dt = dt
        _t = t
        self._p = percent
        self._max_it = math.ceil(_t/_dt)
        self._count = 0
        self._curr_p = percent
        self.timer = ppmd.opt.Timer(runtime.TIMER, 0, start=False)
        self._timing = False

    def tick(self):
        """
        Method to call per iteration.
        """

        if (self._timing is False) and (runtime.TIMER > 0):
            self.timer.start()

        self._count += 1

        if (float(self._count)/self._max_it)*100 > self._curr_p:

            if runtime.TIMER > 0:
                pio.pprint(self._curr_p, "%", self.timer.reset(), 's')
            else:
                pio.pprint(self._curr_p, "%")

            self._curr_p += self._p





class ParticleTracker(object):
    def __init__(self, dat=None, index=None, filename=None):
        """
        Writes the index in a particle dat to a file
        """
        assert dat is not None, "No dat"
        assert index is not None, "No index"
        assert filename is not None, "No filename"


        self._dat = dat
        self._fh = open(filename, 'w')
        self._i = index

    def write(self):
        """
        Call to write at a particular point in time.
        """

        for lx in range(self._dat.ncomp):
            self._fh.write("%(VAL)s\t" % {'VAL':str(self._dat.data[self._i,lx])})
        self._fh.write('\n')


    def finalise(self):
        self._fh.close()
        self._fh = None












