import math
import data
import pairloop
import loop
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

import collections
import kernel
import ctypes
import time
import os
import re
import datetime
import inspect
import build
import data
import runtime
import pio
import mpi
np.set_printoptions(threshold='nan')



###############################################################################################################
# Velocity Verlet Method
###############################################################################################################

class VelocityVerlet(object):
    """
    Class to apply Velocity-Verlet to a given state using a given looping method.
    
    :arg double dt: Time step size, can be specified at integrate call.
    :arg double T: End time, can be specified at integrate call.
    :arg bool USE_C: Flag to use C looping and kernel.
    :arg DrawParticles plot_handle: PLotting class to plot state at certain progress points.
    :arg BasicEnergyStore energy_handle: Energy storage class to log energy at each iteration.
    :arg bool writexyz: Flag to indicate writing of xyz at each DT.
    :arg bool DEBUG: Flag to enable debug flags.
    """
    
    def __init__(self, dt=0.0001, t=0.01, simulation=None, schedule=None):
    
        self._dt = dt
        self._T = t

        self._sim = simulation
        self._state = self._sim.state

        self.timer = runtime.Timer(runtime.TIMER, 0)
        
        self._domain = self._state.domain
        self._N = self._state.n
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
        
        if dt is not None:
            self._dt = dt
        if t is not None:
            self._T = t

        self._max_it = int(math.ceil(self._T/self._dt))

        self._constants = [kernel.Constant('dt',self._dt), kernel.Constant('dht',0.5*self._dt),]

        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
        self._p1 = loop.ParticleLoop(self._N, self._state.types,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M})

        self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
        self._p2 = loop.ParticleLoop(self._N, self._state.types,self._kernel2,{'V':self._V,'A':self._A, 'M':self._M})
        
        
        self._sim.execute_boundary_conditions()
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
            # print mpi.MPI_HANDLE.rank, "-------", i , self._state.n

            self._p1.execute(self._state.n)

            #self._sim.execute_boundary_conditions()

            #TODO: fix this.
            self._sim.forces_update()
            self._p2.execute(self._state.n)

            self._sim.kinetic_energy_update()
            self._state.time += self._dt

            if self._schedule is not None:
                self._schedule.tick()



################################################################################################################
# Anderson thermostat
################################################################################################################


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
        self._p1 = loop.ParticleLoop(self._N, self._state.types ,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M})

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
            const double M_tmp = 1/M(0);
            V(0) += dht*A(0)*M_tmp;
            V(1) += dht*A(1)*M_tmp;
            V(2) += dht*A(2)*M_tmp;
        }

        '''

        self._constants2_thermostat = [kernel.Constant('rate',self._dt*self._nu), kernel.Constant('dt',self._dt), kernel.Constant('dht',0.5*self._dt), kernel.Constant('temperature',self._Temp),]

        self._kernel2_thermostat = kernel.Kernel('vv2_thermostat',self._kernel2_thermostat_code,self._constants2_thermostat, headers = ['math.h','stdlib.h','time.h','stdio.h'])
        self._p2_thermostat = loop.ParticleLoop(self._N, self._state.types, self._kernel2_thermostat,{'V':self._V,'A':self._A, 'M':self._M})

        _t = runtime.Timer(runtime.TIMER, 0, start=True)
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

                
################################################################################################################
# G(R)
################################################################################################################


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
        self._N = self._state.n
        self._rmax = rmax
        
        if self._rmax is None:
            self._rmax = 0.5*np.min(self._extent.dat)
        
        
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

        self._p = pairloop.DoubleAllParticleLoop(self._N, self._state.types, kernel=_grkernel, particle_dat_dict=_datdict)

        self.timer = runtime.Timer(runtime.TIMER, 0)

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
        self._grscaled = self._gr.dat*self._state.domain.volume/((self._N())*(self._N() - 1)*2*math.pi*(self._r**2) * (self._rmax/float(self._rsteps))*self._count)
        
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
                print "Warning: run evaluate() at least once before plotting."

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


################################################################################################################
# WriteTrajectoryXYZ
################################################################################################################


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

        if mpi.MPI_HANDLE.rank == 0:
            self._fh = open(os.path.join(self._dn, self._fn), 'w')
            self._fh.close()

        self.timer = runtime.Timer(runtime.TIMER, 0)

    def write(self):
        """
        Append current positions to file.
        :return:
        """

        self.timer.start()

        space = ' '

        if mpi.MPI_HANDLE.rank == 0:
            self._fh = open(os.path.join(self._dn, self._fn), 'a')
            self._fh.write(str(self._s.nt) + '\n')
            self._fh.write(str(self._title) + '\n')
            self._fh.flush()
        mpi.MPI_HANDLE.barrier()

        if self._ordered is False:
            for iz in range(mpi.MPI_HANDLE.nproc):
                if mpi.MPI_HANDLE.rank == iz:
                    self._fh = open(os.path.join(self._dn, self._fn), 'a')
                    for ix in range(self._s.n):
                        self._fh.write(str(self._symbol).rjust(3))
                        for iy in range(3):
                            self._fh.write(space + str('%.5f' % self._s.positions[ix, iy]))
                        self._fh.write('\n')

                    self._fh.flush()
                    self._fh.close()

                mpi.MPI_HANDLE.barrier()

        self.timer.pause()

################################################################################################################
# Schedule
################################################################################################################


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
                print "Schedule warning: 0 step items will be ignored."
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
        self._V0 = data.ParticleDat(self._state.n, 3, name='v0')
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

        self._loop = loop.ParticleLoop(self._state.as_func('n'), None, kernel=_kernel, particle_dat_dict=self._datdict)

    def set_v0(self, v0=None, state=None):
        """
        Set an initial velocity Dat to use as V_0. Requires either a velocity Dat or a state as an argument. V_0 will be set to either the passed velocities or to the velocities in the passed state.

        :arg data.ParticleDat v0: Velocity Dat.
        :arg state state: State class containing velocities.
        """

        if v0 is not None:
            self._V0.dat = np.copy(v0.dat)
            self._V0_SET = True
        if state is not None:
            self._V0.dat = np.copy(state.velocities.dat)
            self._V0_SET = True
        assert self._V0_SET is True, "No velocities set, check input data."


    def evaluate(self):
        """
        Evaluate VAF using the current velocities held in the state with the velocities in V0.

        :arg double t: Time within block of integration.
        """
        if runtime.TIMER.level > 0:
            start = time.time()

        _t = self._state.time

        _Ni = 1./self._state.as_func('n')

        self._datdict['VT'] = self._state.velocities
        self._loop.execute(None, self._datdict, {'Ni': ctypes.c_double(_Ni)})

        self._V.append(self._VAF[0])
        self._T.append(_t)

        if runtime.TIMER.level > 0:
            end = time.time()
            pio.pprint("VAF time taken:", end - start, "s")

    def plot(self):
        """
        Plot array of recorded VAF evaluations.
        """

        if _GRAPHICS:

            _Vloc = np.array(self._V)
            _V = np.zeros(len(self._T))

            print _Vloc

            mpi.MPI_HANDLE.comm.Reduce(_Vloc, _V, data.MPI.SUM, 0)

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
                print "Warning: run evaluate() at least once before plotting."


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

        self._Mh = mpi.MPI_HANDLE

        self._Dat = None
        self._gids = None
        self._pos = None
        self._gid = None

        self._N = None
        self._NT = None
        self._extents = None

        if (mpi.MPI_HANDLE.rank == 0) and _GRAPHICS:
            plt.ion()
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._key = ['red', 'blue']
            plt.show(block=False)

    def draw(self):
        """
        Update current plot, use for real time plotting.
        """

        if _GRAPHICS:

            self._N = self._state.n
            self._NT = self._state.nt
            self._extents = self._state.domain.extent

            '''Case where all particles are local'''
            if self._Mh is None:
                self._pos = self._state.positions
                self._gid = self._state.global_ids

            else:
                '''Need an mpi handle if not all particles are local'''
                assert self._Mh is not None, "Error: Not all particles are local but mpi.MPI_HANDLE = None."

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

                if self._Mh.rank == 0:

                    '''Copy the local data.'''
                    self._Dat.dat[0:self._N:, ::] = self._state.positions.dat[0:self._N:, ::]
                    self._gids.dat[0:self._N:] = self._state.global_ids.dat[0:self._N:, 0]

                    _i = self._N  # starting point pos
                    _ig = self._N  # starting point gids

                    for ix in range(1, self._Mh.nproc):
                        self._Mh.comm.Recv(self._Dat.dat[_i::, ::], ix, ix, _MS)
                        _i += _MS.Get_count(mpi.mpi_map[self._Dat.dtype]) / 3

                        self._Mh.comm.Recv(self._gids.dat[_ig::], ix, ix, _MS)
                        _ig += _MS.Get_count(mpi.mpi_map[self._gids.dtype])

                    self._pos = self._Dat
                    self._gid = self._gids
                else:

                    self._Mh.comm.Send(self._state.positions.dat[0:self._N:, ::], 0, self._Mh.rank)
                    self._Mh.comm.Send(self._state.global_ids.dat[0:self._N:], 0, self._Mh.rank)

            if self._Mh.rank == 0:

                plt.cla()
                plt.ion()
                for ix in range(self._pos.npart):
                    self._ax.scatter(self._pos.dat[ix, 0], self._pos.dat[ix, 1], self._pos.dat[ix, 2],
                                     color=self._key[self._gid[ix] % 2])
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
        self._Mh = mpi.MPI_HANDLE

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

        if self._state.n > 0:
            # print self._state.u[0], self._state.u[1]

            _U_tmp = self._state.u.dat[0]/self._state.nt
            _U_tmp += 0.5*self._state.u.dat[1]/self._state.nt
            _u = _U_tmp

            _k = self._state.k[0]/self._state.nt
            _q = _U_tmp+(self._state.k[0])/self._state.nt


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

        if (self._Mh is not None) and (self._Mh.nproc > 1):

            # data to collect
            _d = [self._Q_store.dat, self._U_store.dat, self._K_store.dat]

            # make a temporary buffer.
            if self._Mh.rank == 0:
                _buff = data.ScalarArray(ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _T = self._T_store.dat
                _Q = data.ScalarArray( ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _U = data.ScalarArray(ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _K = data.ScalarArray(ncomp=self._T_store.ncomp, dtype=ctypes.c_double)

                _Q.dat[::] += self._Q_store.dat[::]
                _U.dat[::] += self._U_store.dat[::]
                _K.dat[::] += self._K_store.dat[::]

                _dl = [_Q.dat, _U.dat, _K.dat]
            else:
                _dl = [None, None, None]

            for _di, _dj in zip(_d, _dl):

                if self._Mh.rank == 0:
                    _MS = mpi.Status()
                    for ix in range(1, self._Mh.nproc):
                        self._Mh.comm.Recv(_buff.dat[::], ix, ix, _MS)
                        _dj[::] += _buff.dat[::]

                else:
                    self._Mh.comm.Send(_di[::], 0, self._Mh.rank)

            if self._Mh.rank == 0:
                _Q = _Q.dat
                _U = _U.dat
                _K = _K.dat

        else:
            _T = self._T_store.dat
            _Q = self._Q_store.dat
            _U = self._U_store.dat
            _K = self._K_store.dat



        if (mpi.MPI_HANDLE.rank == 0) and _GRAPHICS:
            print "last total", _Q[-1]
            print "last kinetic", _K[-1]
            print "last potential", _U[-1]
            print "============================================="
            print "first total", _Q[0]
            print "first kinetic", _K[0]
            print "first potential", _U[0]

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

        if mpi.MPI_HANDLE.rank == 0:
            if not os.path.exists(os.path.join(os.getcwd(),'./output')):
                os.system('mkdir ./output')


            _fh = open('./output/energy.txt', 'w')
            _fh.write("Time Kinetic Potential Total\n")
            for ix in range(len(self._t)):
                _fh.write("%(T)s %(K)s %(P)s %(Q)s\n" % {'T':_T[ix], 'K':_K[ix], 'P':_U[ix], 'Q':_Q[ix]})
            _fh.close()

        if (mpi.MPI_HANDLE.rank == 0) and not _GRAPHICS:
            print "last total", _Q[-1]
            print "last kinetic", _K[-1]
            print "last potential", _U[-1]
            print "============================================="
            print "first total", _Q[0]
            print "first kinetic", _K[0]
            print "first potential", _U[0]

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
        self.timer = runtime.Timer(runtime.TIMER, 0, start=False)
        self._timing = False

    def tick(self):
        """
        Method to call per iteration.
        """

        if (self._timing is False) and (runtime.TIMER.level > 0):
            self.timer.start()

        self._count += 1

        if (float(self._count)/self._max_it)*100 > self._curr_p:

            if runtime.TIMER.level > 0:
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
            self._fh.write("%(VAL)s\t" % {'VAL':str(self._dat.dat[self._i,lx])})
        self._fh.write('\n')


    def finalise(self):
        self._fh.close()
        self._fh = None












