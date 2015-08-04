import math
import data
import pairloop
import loop
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import kernel
import constant
import ctypes
import time
import os
import re
import datetime
import particle
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
    
    def __init__(self, dt=0.0001, t=0.01, dt_step=0.001, state=None, plot_handle = None, energy_handle = None, writexyz = False, VAF_handle = None, DEBUG = False, MPI_handle = None):
    
        self._dt = dt
        self._DT = dt_step
        self._T = t
        self._DEBUG = DEBUG
        self._Mh = MPI_handle

        self._state = state
        
        self._domain = self._state.domain
        self._N = self._state.n
        self._A = self._state.forces
        self._V = self._state.velocities
        self._P = self._state.positions
        self._M = self._state.masses
        self._K = self._state.k
        

        self._plot_handle = plot_handle
        self._energy_handle = energy_handle
        self._writexyz = writexyz
        self._VAF_handle = VAF_handle
        
        self._kernel1_code = '''
        //self._V+=0.5*self._dt*self._A
        //self._P+=self._dt*self._V
        const double M_tmp = 1/M[0];
        V[0] += dht*A[0]*M_tmp;
        V[1] += dht*A[1]*M_tmp;
        V[2] += dht*A[2]*M_tmp;
        P[0] += dt*V[0];
        P[1] += dt*V[1];
        P[2] += dt*V[2];
        '''
                
        self._kernel2_code = '''
        //self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat
        const double M_tmp = 1/M[0];
        V[0] += dht*A[0]*M_tmp;
        V[1] += dht*A[1]*M_tmp;
        V[2] += dht*A[2]*M_tmp;
        '''
        
        self._K_kernel_code = '''
        
        k[0] += (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];
        
        '''      
        self._constants_K = []
        self._K_kernel = kernel.Kernel('K_kernel',self._K_kernel_code,self._constants_K)
        self._pK = loop.SingleAllParticleLoop(self._N, self._state.types_map, self._K_kernel, {'V':self._V,'k':self._K, 'M':self._M}, DEBUG = self._DEBUG, mpi_handle = self._Mh)
        
               
        
    def integrate(self, dt = None, DT = None, T = None, timer=False):
        """
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double T: End time.
        :arg bool timer: display approximate timing information.
        """

        
        
        if (dt != None):
            self._dt = dt
        if (T != None):
            self._T = T
        if (DT != None):
            self._DT = DT
        else:
            self._DT = 10.0*self._dt
            
        self._max_it = int(math.ceil(self._T/self._dt))
        self._DT_Count = int(math.ceil(self._T/self._DT))
        
        if self._energy_handle is not None:
            self._energy_handle.append_prepare(self._DT_Count)
        
        if self._VAF_handle is not None:
            self._VAF_handle.append_prepare(self._DT_Count)


        self._constants = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]

        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
        self._p1 = loop.SingleAllParticleLoop(self._N, self._state.types_map,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG, mpi_handle=self._Mh)

        self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
        self._p2 = loop.SingleAllParticleLoop(self._N, self._state.types_map,self._kernel2,{'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG, mpi_handle=self._Mh)
              
        if timer is True:
            start = time.time()        
        self._velocity_verlet_integration()
        if timer is True:
            end = time.time()
            if self._state.domain.rank == 0:
                print "integrate time taken:", end - start, "s"
        
        
        
        
    def _velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        self._percent_count = 101
        if self._plot_handle is not None:
            self._percent_int = self._plot_handle.interval
            self._percent_count = self._percent_int

        # print "before bc"
        self._domain.bc_execute()
        
        # print "after bc"
        
        self._state.forces_update()

        for i in range(self._max_it):

            self._velocity_verlet_step(self._state.n())
            
            self._integration_internals(i)

    def _velocity_verlet_step(self, N):
        """
        Perform one step of Velocity Verlet.
        """

        self._p1.execute(N)
        
        # update forces

        self._domain.bc_execute()
        self._state.forces_update()
        self._p2.execute(N)

    def _integration_internals(self, i):

        DTFLAG = ( ((i + 1) % (self._max_it/self._DT_Count) == 0) | (i == (self._max_it-1)) )
        PERCENT = ((100.0*i)/self._max_it)
        
        if (self._energy_handle is not None) & (DTFLAG is True):

            self._K.scale(0.0)
            self._pK.execute()

            if self._N() > 0:
                
                _U_tmp = self._state.u.dat[0]/self._state.nt()

                _U_tmp += 0.5*self._state.u.dat[1]/self._state.nt()
                
                self._energy_handle.u_append(_U_tmp)

                if self._N() > 0:
                    self._energy_handle.k_append((self._K[0])/self._state.nt())
                    self._energy_handle.q_append(_U_tmp + (self._K[0])/self._state.nt())
                else:
                    self._energy_handle.k_append(0.)
                    self._energy_handle.q_append(0.)

            else:
                self._energy_handle.u_append(0.)
                self._energy_handle.k_append(0.)
                self._energy_handle.q_append(0.)
            self._energy_handle.t_append((i+1)*self._dt)

        if (self._writexyz is True) & (DTFLAG is True):
            self._state.positions.XYZWrite(append=1)

        if (self._VAF_handle is not None) & (DTFLAG is True):
            self._VAF_handle.evaluate(t=(i+1)*self._dt)

        if (self._plot_handle is not None) & (PERCENT > self._percent_count):
            
            if self._plot_handle is not None:
                self._plot_handle.draw(self._state)
            
            self._percent_count += self._percent_int
            
            if self._state.domain.rank == 0:
                print int((100.0*i)/self._max_it),"%", "T=", self._dt*i
                

################################################################################################################
# Anderson thermostat
################################################################################################################


class VelocityVerletAnderson(VelocityVerlet):
    
    def integrate_thermostat(self, dt=None, dt_step=None, t=None, temp=273.15, nu=1.0, timer=False):
        """
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double t: End time.
        :arg double temp: Temperature of heat bath.
        :arg bool timer: display approximate timing information.
        """
        if timer:
            start = time.time()
        
        self._Temp = temp
        self._nu = nu
        
        if dt is not None:
            self._dt = dt
        if t is not None:
            self._T = t
        if dt_step is not None:
            self._DT = dt_step
        else:
            self._DT = 10.0*self._dt
            
        self._max_it = int(math.ceil(self._T/self._dt))
        self._DT_Count = int(math.ceil(self._T/self._DT))
        if self._energy_handle is not None:
            self._energy_handle.append_prepare(self._DT_Count)
            
        if self._VAF_handle is not None:
            self._VAF_handle.append_prepare(self._DT_Count)

        self._constants1 = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants1)
        self._p1 = loop.SingleAllParticleLoop(self._N, self._state.types_map ,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG, mpi_handle = self._Mh)

        self._kernel2_thermostat_code = '''

        //Anderson thermostat here.
        //probably horrific random code.

        const double tmp_rand_max = 1.0/RAND_MAX;

        if (rand()*tmp_rand_max < rate) {

            //Box-Muller method.


            const double scale = sqrt(temperature/M[0]);
            const double stmp = scale*sqrt(-2.0*log(rand()*tmp_rand_max));

            const double V0 = 2.0*M_PI*rand()*tmp_rand_max;
            V[0] = stmp*cos(V0);
            V[1] = stmp*sin(V0);
            V[2] = scale*sqrt(-2.0*log(rand()*tmp_rand_max))*cos(2.0*M_PI*rand()*tmp_rand_max);

        }
        else {
            const double M_tmp = 1/M[0];
            V[0] += dht*A[0]*M_tmp;
            V[1] += dht*A[1]*M_tmp;
            V[2] += dht*A[2]*M_tmp;
        }

        '''


        self._constants2_thermostat = [constant.Constant('rate',self._dt*self._nu), constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt), constant.Constant('temperature',self._Temp),]

        self._kernel2_thermostat = kernel.Kernel('vv2_thermostat',self._kernel2_thermostat_code,self._constants2_thermostat, headers = ['math.h','stdlib.h','time.h','stdio.h'])
        self._p2_thermostat = loop.SingleAllParticleLoop(self._N, self._state.types_map, self._kernel2_thermostat,{'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG, mpi_handle = self._Mh)

        
        self._velocity_verlet_integration_thermostat()
        
        if (timer is True) and (self._Mh.rank == 0):
            end = time.time()
            print "integrate thermostat time taken:", end - start, "s"
    
    def _velocity_verlet_integration_thermostat(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        
        self._percent_count = 101
        if self._plot_handle is not None:
            self._percent_int = self._plot_handle.interval
            self._percent_count = self._percent_int

        self._domain.bc_execute()
        self._state.forces_update()

        for i in range(self._max_it):
              
            self._p1.execute()
            
            # update forces
            self._domain.bc_execute()
            self._state.forces_update()
            
            self._p2_thermostat.execute() 
            
            self._integration_internals(i)


################################################################################################################
# SOLID BOUNDARY TEST INTEGRATOR
################################################################################################################


class VelocityVerletBox(VelocityVerlet):
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
    
    def __init__(self, dt=0.0001, t=0.01, dt_step=0.001, state=None, plot_handle=None, energy_handle=None, writexyz=False, vaf_handle=None, DEBUG=False, mpi_handle=None, writer = None):
    
        self._dt = dt
        self._DT = dt_step
        self._T = t
        self._DEBUG = DEBUG
        self._Mh = mpi_handle

        self._state = state

        self._writer = writer
        
        self._domain = self._state.domain
        self._N = self._state.n
        self._A = self._state.forces
        self._V = self._state.velocities
        self._P = self._state.positions
        self._M = self._state.masses
        self._K = self._state.k

        self._plot_handle = plot_handle
        self._energy_handle = energy_handle
        self._writexyz = writexyz
        self._VAF_handle = vaf_handle
        
        self._kernel1_code = '''
        //self._V+=0.5*self._dt*self._A
        //self._P+=self._dt*self._V
        const double M_tmp = 1/M[0];
        V[0] += dht*A[0]*M_tmp;
        V[1] += dht*A[1]*M_tmp;
        V[2] += dht*(A[2]-100.0)*M_tmp;
        P[0] += dt*V[0];
        P[1] += dt*V[1];
        P[2] += dt*V[2];
        
        if (P[0] > 0.5*E[0]){ P[0] = 0.5*E[0]-0.000001; V[0]=-0.8*V[0]; }
        if (P[1] > 0.5*E[1]){ P[1] = 0.5*E[1]-0.000001; V[1]=-0.8*V[1]; }
        if (P[2] > 0.5*E[2]){ P[2] = 0.5*E[2]-0.000001; V[2]=-0.8*V[2]; }
        
        if (P[0] < -0.5*E[0]){ P[0] = -0.5*E[0]+0.000001; V[0]=-0.8*V[0]; }
        if (P[1] < -0.5*E[1]){ P[1] = -0.5*E[1]+0.000001; V[1]=-0.8*V[1]; }
        if (P[2] < -0.5*E[2]){ P[2] = -0.5*E[2]+0.000001; V[2]=-0.8*V[2]; }        
        
        
        '''
                
        self._kernel2_code = '''
        //self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat
        const double M_tmp = 1/M[0];
        V[0] += dht*A[0]*M_tmp;
        V[1] += dht*A[1]*M_tmp;
        V[2] += dht*(A[2]-100.0)*M_tmp;
        '''
        
        self._K_kernel_code = '''
        
        k[0] += (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];
        
        '''      
        self._constants_K = []
        self._K_kernel = kernel.Kernel('K_kernel', self._K_kernel_code, self._constants_K)
        self._pK = loop.SingleAllParticleLoop(self._N,
                                              self._state.types_map,
                                              self._K_kernel,
                                              {'V': self._V, 'k': self._K, 'M': self._M, 'E': self._domain.extent},
                                              DEBUG=self._DEBUG,
                                              mpi_handle=self._Mh)
    
    def integrate(self, dt=None, dt_step=None, t=None, timer=False):
        """
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double t: End time.
        :arg bool timer: display approximate timing information.
        """

        if dt is not None:
            self._dt = dt
        if t is not None:
            self._T = t
        if dt_step is not None:
            self._DT = dt_step
        else:
            self._DT = 10.0*self._dt
            
        self._max_it = int(math.ceil(self._T/self._dt))
        self._DT_Count = int(math.ceil(self._T/self._DT))
        
        if self._energy_handle is not None:
            self._energy_handle.append_prepare(self._DT_Count)
        
        if self._VAF_handle is not None:
            self._VAF_handle.append_prepare(self._DT_Count)
        
        self._constants = [constant.Constant('dt', self._dt), constant.Constant('dht', 0.5*self._dt), ]

        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
        self._p1 = loop.SingleAllParticleLoop(self._N, self._state.types_map,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M, 'E':self._domain.extent}, DEBUG = self._DEBUG, mpi_handle = self._Mh)

        self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
        self._p2 = loop.SingleAllParticleLoop(self._N, self._state.types_map,self._kernel2,{'V':self._V,'A':self._A, 'M':self._M, 'E':self._domain.extent}, DEBUG = self._DEBUG, mpi_handle = self._Mh)


        if timer:
            start = time.time()        
        self._velocity_verlet_integration()
        if timer:
            end = time.time()
            if self._state.domain.rank == 0:
                print "integrate time taken:", end - start, "s"

    def integrate_thermostat(self, dt=None, dt_step=None, t=None, temp=273.15, nu=1.0, timer=False):
        """
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double t: End time.
        :arg double temp: Temperature of heat bath.
        :arg bool timer: display approximate timing information.
        """

        if timer:
            start = time.time()
        
        self._Temp = temp
        self._nu = nu
        
        if dt is not None:
            self._dt = dt
        if t is not None:
            self._T = t
        if dt_step is not None:
            self._DT = dt_step
        else:
            self._DT = 10.0*self._dt
            
        self._max_it = int(math.ceil(self._T/self._dt))
        self._DT_Count = int(math.ceil(self._T/self._DT))

        if self._energy_handle is not None:
            self._energy_handle.append_prepare(self._DT_Count)
            
        if self._VAF_handle is not None:
            self._VAF_handle.append_prepare(self._DT_Count)

        self._constants1 = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
        self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants1)
        self._p1 = loop.SingleAllParticleLoop(self._N, self._state.types_map ,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M, 'E':self._domain.extent}, DEBUG = self._DEBUG, mpi_handle = self._Mh)

        self._kernel2_thermostat_code = '''

        //Anderson thermostat here.
        //probably horrific random code.

        const double tmp_rand_max = 1.0/RAND_MAX;

        if (rand()*tmp_rand_max < rate) {

            //Box-Muller method.


            const double scale = sqrt(temperature/M[0]);
            const double stmp = scale*sqrt(-2.0*log(rand()*tmp_rand_max));

            const double V0 = 2.0*M_PI*rand()*tmp_rand_max;
            V[0] = stmp*cos(V0);
            V[1] = stmp*sin(V0);
            V[2] = scale*sqrt(-2.0*log(rand()*tmp_rand_max))*cos(2.0*M_PI*rand()*tmp_rand_max);

        }
        else {
            const double M_tmp = 1/M[0];
            V[0] += dht*A[0]*M_tmp;
            V[1] += dht*A[1]*M_tmp;
            V[2] += dht*(A[2]-100.0)*M_tmp;
        }

        '''

        self._constants2_thermostat = [constant.Constant('rate',self._dt*self._nu), constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt), constant.Constant('temperature',self._Temp),]

        self._kernel2_thermostat = kernel.Kernel('vv2_thermostat',self._kernel2_thermostat_code,self._constants2_thermostat, headers = ['math.h','stdlib.h','time.h','stdio.h'])
        self._p2_thermostat = loop.SingleAllParticleLoop(self._N, self._state.types_map, self._kernel2_thermostat,{'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG, mpi_handle = self._Mh)

        
        self._velocity_verlet_integration_thermostat()
        
        if (timer is True) and (self._Mh.rank == 0):
            end = time.time()
            print "integrate thermostat time taken:", end - start, "s"
    
    def _velocity_verlet_integration_thermostat(self):
        """
        Perform Velocity Verlet integration up to time T.
        """

        self._percent_count = 101
        if self._plot_handle is not None:
            self._percent_int = self._plot_handle.interval
            self._percent_count = self._percent_int

        self._domain.bc_execute()
        self._state.forces_update()

        for i in range(self._max_it):
              
            self._p1.execute()
            
            # update forces
            self._domain.bc_execute()
            self._state.forces_update()
            
            self._p2_thermostat.execute() 
            
            self._integration_internals(i)

    def _integration_internals(self, i):

        DTFLAG = ( ((i + 1) % (self._max_it/self._DT_Count) == 0) | (i == (self._max_it-1)) )
        PERCENT = ((100.0*i)/self._max_it)



        if self._writer is not None:
            self._writer()

        if (self._energy_handle is not None) & (DTFLAG is True):

            self._K.scale(0.0)
            self._pK.execute()

            if self._N() > 0:

                _U_tmp = self._state.u.dat[0]/self._state.nt()

                _U_tmp += 0.5*self._state.u.dat[1]/self._state.nt()

                self._energy_handle.u_append(_U_tmp)

                if self._N() > 0:
                    self._energy_handle.k_append((self._K[0])/self._state.nt())
                    self._energy_handle.q_append(_U_tmp + (self._K[0])/self._state.nt())
                else:
                    self._energy_handle.k_append(0.)
                    self._energy_handle.q_append(0.)

            else:
                self._energy_handle.u_append(0.)
                self._energy_handle.k_append(0.)
                self._energy_handle.q_append(0.)
            self._energy_handle.t_append((i+1)*self._dt)

        if (self._writexyz is True) & (DTFLAG is True):
            self._state.positions.XYZWrite(append=1)

        if (self._VAF_handle is not None) & (DTFLAG is True):
            self._VAF_handle.evaluate(t=(i+1)*self._dt)

        if (self._plot_handle is not None) & (PERCENT > self._percent_count):

            if self._plot_handle is not None:
                self._plot_handle.draw(self._state)

            self._percent_count += self._percent_int

            if self._state.domain.rank == 0:
                print int((100.0*i)/self._max_it),"%", "T=", self._dt*i
                
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

    def __init__(self, state, rmax=None, rsteps=100, DEBUG=False, mpi_handle=None):
        
        self._count = 0
        self._state = state
        self._extent = self._state.domain.extent
        self._P = self._state.positions
        self._N = self._state.n
        self._rmax = rmax
        
        if self._rmax is None:
            self._rmax = 0.5*self._extent.min
        
        
        self._rsteps = rsteps
        
        self._gr = data.ScalarArray(ncomp=self._rsteps, dtype=ctypes.c_int)
        self._gr.scale(0.0)
        
        self._DEBUG = DEBUG
        self._Mh = mpi_handle
        
        _headers = ['math.h', 'stdio.h']
        _kernel = '''
        
        
        double R0 = P[1][0] - P[0][0];
        double R1 = P[1][1] - P[0][1];
        double R2 = P[1][2] - P[0][2];
        
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
        
        _constants=(constant.Constant('rmaxoverrsteps', 0.2*self._rmax/self._rsteps ),
                    constant.Constant('rstepsoverrmax', self._rsteps/self._rmax ),
                    constant.Constant('rmax2', self._rmax**2 ),
                    constant.Constant('extent0', self._extent[0] ),
                    constant.Constant('extent1', self._extent[1] ),
                    constant.Constant('extent2', self._extent[2] ),
                    constant.Constant('exto20', 0.5*self._extent[0] ),
                    constant.Constant('exto21', 0.5*self._extent[1] ),
                    constant.Constant('exto22', 0.5*self._extent[2] )
                    )

        _grkernel = kernel.Kernel('radial_distro_periodic_static', _kernel, _constants, headers=_headers)
        _datdict = {'P':self._P, 'GR':self._gr}

        self._p = pairloop.DoubleAllParticleLoop(self._N, self._state.types_map, kernel=_grkernel, particle_dat_dict=_datdict, DEBUG=self._DEBUG, mpi_handle=self._Mh)
        
    def evaluate(self, timer=False):
        """
        Evaluate the radial distribution function.
        
        :arg bool timer: display approximate timing information.
        """
        
        assert self._rmax <= 0.5*self._state.domain.extent.min, "Maximum radius too large."

        if timer:
            start = time.time()    
        self._p.execute()
        self._count += 1

        if timer:
            end = time.time()
            print "rdf time taken:", end - start, "s"
        
    def _scale(self):
        self._r = np.linspace(0.+0.5*(self._rmax/self._rsteps), self._rmax-0.5*(self._rmax/self._rsteps), num=self._rsteps, endpoint=True)
        self._grscaled = self._gr.dat*self._state.domain.volume/((self._N())*(self._N() - 1)*2*math.pi*(self._r**2) * (self._rmax/float(self._rsteps))*self._count)
        
    def plot(self):
        self._scale()
        if (self._count > 0):
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
# VAF
################################################################################################################


class VelocityAutoCorrelation(object):
    """
    Method to calculate Velocity Autocorrelation Function.
    
    :arg state state: Input state containing velocities.
    :arg int size: Initial length of VAF array (optional).
    :arg particle.Dat V0: Initial velocity Dat (optional).
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, state, size=0, v0=None, DEBUG=False, mpi_handle=None):
        self._DEBUG = DEBUG
        self._Mh = mpi_handle
        self._state = state
        self._N = self._state.n
        self._V0 = particle.Dat(self._N, 3, name='v0')
        self._VT = self._V0

        self._VO_SET = False
        if v0 is not None:
            self.set_v0(v0)
        else:
            self.set_v0(state = self._state)

        self._VAF = data.ScalarArray(ncomp=size)
        self._VAF_index = 0
        
        self._T_store = data.ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._T_base = None
        
        _headers = ['stdio.h']
        _constants = None
        _kernel_code = '''
        
        VAF[I] += (v0[0]*VT[0] + v0[1]*VT[1] + v0[2]*VT[2])*Ni;
        
        '''
        _reduction = (kernel.Reduction('VAF', 'VAF[I]', '+'),)
        
        _static_args = {'I': ctypes.c_int, 'Ni': ctypes.c_double}
        
        _kernel = kernel.Kernel('VelocityAutocorrelation', _kernel_code, _constants, _headers, _reduction, _static_args)

        self._datdict = {'VAF': self._VAF, 'v0': self._V0, 'VT': self._VT}
        
        self._loop = loop.SingleAllParticleLoop(self._N, None, kernel=_kernel, particle_dat_dict=self._datdict, DEBUG=self._DEBUG, mpi_handle=self._Mh)

    def set_v0(self, v0=None, state=None):
        """
        Set an initial velocity Dat to use as V_0. Requires either a velocity Dat or a state as an argument. V_0 will be set to either the passed velocities or to the velocities in the passed state.        
        
        :arg particle.Dat v0: Velocity Dat.
        :arg state state: State class containing velocities.
        """
        
        if v0 is not None:
            self._V0.dat = np.copy(v0.dat)
            self._V0_SET = True
        if state is not None:
            self._V0.dat = np.copy(state.velocities.dat)
            self._V0_SET = True            
        assert self._V0_SET is True, "No velocities set, check input data."

    def evaluate(self, t=None, timer=False):
        """
        Evaluate VAF using the current velocities held in the state with the velocities in V0.
        
        :arg double t: Time within block of integration.
        :arg bool timer: Flag to time evaluation of VAF.
        """
        if timer:
            start = time.time()

        assert int(self._VAF_index) < int(self._VAF.ncomp), "VAF store not large enough"
        
        self._Ni = 1./self._N()
        self._datdict['VT'] = self._state.velocities     
        self._loop.execute(None, self._datdict, {'I':(ctypes.c_int)(self._VAF_index),'Ni':(ctypes.c_double)(self._Ni)})
        
        if t is None:
            self._T_store[self._VAF_index] = 1 + self._T_base
        else:
            
            self._T_store[self._VAF_index] = t + self._T_base
        
        self._VAF_index += 1

        if timer:
            end = time.time()
            print "VAF time taken:", end - start,"s"         

    def append_prepare(self,size):
        """
        Function to prepare storage arrays for forthcoming VAF evaluations.
        
        :arg int size: Number of upcoming evaluations.
        """
        self._VAF.concatenate(size)      
        
        if self._T_base is  None:
            self._T_base = 0.0
        else:
            self._T_base = self._T_store[-1]        
        self._T_store.concatenate(size)

    def plot(self):
        """
        Plot array of recorded VAF evaluations.
        """
        
        if self._VAF_index > 0:
            plt.ion()
            _fig = plt.figure()
            _ax = _fig.add_subplot(111)
            
            plt.plot(self._T_store.dat, self._VAF.dat)
            _ax.set_title('Velocity Autocorrelation Function')
            _ax.set_xlabel('Time')
            _ax.set_ylabel('VAF')
            plt.show()
        else:
            print "Warning: run evaluate() at least once before plotting."




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
                assert os.path.exists(os.path.join(dir_name, file_name)), "WriteTrajectoryXYZ Error: No unique name found."

        self._fn = file_name
        self._dn = dir_name
        self._fh = None

        if self._s.mpi_handle.rank == 0:
            self._fh = open(os.path.join(self._dn, self._fn), 'w')
            self._fh.close()

    def write(self):
        """
        Append current positions to file.
        :return:
        """
        space = ' '



        if self._s.mpi_handle.rank == 0:
            self._fh = open(os.path.join(self._dn, self._fn), 'a')
            self._fh.write(str(self._s.nt()) + '\n')
            self._fh.write(str(self._title) + '\n')
            self._fh.flush()
        self._s.mpi_handle.barrier()

        if self._ordered is False:
            for iz in range(self._s.mpi_handle.nproc):
                if self._s.mpi_handle.rank == iz:
                    self._fh = open(os.path.join(self._dn, self._fn), 'a')
                    for ix in range(self._s.n()):
                        self._fh.write(str(self._symbol).rjust(3))
                        for iy in range(3):
                            self._fh.write(space + str('%.5f' % self._s.positions[ix, iy]))
                        self._fh.write('\n')

                    self._fh.flush()
                    self._fh.close()

                self._s.mpi_handle.barrier()


################################################################################################################
# Schedule
################################################################################################################


class Schedule(object):
    """
    Class to schedule automated running of functions every set number of steps.

    :arg list steps: List of steps between each run.
    :arg list items: list of functions to run after set number of steps.
    """
    
    def __init__(self, steps=None, items=None):
        self._s = collections.defaultdict(list)
        self._sl = []

        if (steps is not None) and (items is not None):
            assert len(steps) == len(items), "Schedule error, mis-match between number of steps and number of items."
            for ix in zip(steps, items):
                self._s[ix[0]].append(ix[1])
                self._sl.append(ix[0])

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
            self._s[ix[0]].append(ix[1])
            self._sl.append(ix[0])

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

        for ix in self._sl:
            if self._count % ix == 0:
                for iy in self._s[ix]:
                    iy()













    
