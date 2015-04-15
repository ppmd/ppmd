import math
import state
import data
import pairloop
import loop
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kernel
import constant
import ctypes
import time
import os
import re
import datetime
import particle
np.set_printoptions(threshold='nan')


################################################################################################################
# Velocity Verlet Method
################################################################################################################ 

class VelocityVerlet(object):
    '''
    Class to apply Velocity-Verlet to a given state using a given looping method.
    
    :arg double dt: Time step size, can be specified at integrate call.
    :arg double T: End time, can be specified at integrate call.
    :arg bool USE_C: Flag to use C looping and kernel.
    :arg DrawParticles plot_handle: PLotting class to plot state at certain progress points.
    :arg BasicEnergyStore energy_handle: Energy storage class to log energy at each iteration.
    :arg bool writexyz: Flag to indicate writing of xyz at each DT.
    :arg bool DEBUG: Flag to enable debug flags.
    '''
    
    def __init__(self, dt = 0.0001, T = 0.01, DT = 0.001,state = None, USE_C = True, plot_handle = None, energy_handle = None, writexyz = False, VAF_handle = None, DEBUG = False):
    
        self._dt = dt
        self._DT = DT
        self._T = T
        self._DEBUG = DEBUG

        self._state = state
        
        self._domain = self._state.domain
        self._N = self._state.N
        self._A = self._state.forces
        self._V = self._state.velocities
        self._P = self._state.positions
        self._M = self._state.masses
        self._K = self._state.K
        
        
        self._USE_C = USE_C
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
        
        K[0] += (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];
        
        '''      
        self._constants_K = []
        self._K_kernel = kernel.Kernel('K_kernel',self._K_kernel_code,self._constants_K)
        self._pK = loop.SingleAllParticleLoop(self._N,self._K_kernel,{'V':self._V,'K':self._K, 'M':self._M}, DEBUG = self._DEBUG)         
        
               
        
    def integrate(self, dt = None, DT = None, T = None, timer=False):
        '''
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double T: End time.
        :arg bool timer: display approximate timing information.
        '''
        if (timer==True):
            start = time.clock()
        
        
        
        if (dt != None):
            self._dt = dt
        if (T != None):
            self._T = T
        if (DT != None):
            self._DT = DT
        else:
            self._DT = 50.0*self._dt
            
        self._max_it = int(math.ceil(self._T/self._dt))
        self._DT_Count = int(math.ceil(self._T/self._DT))
        
        if (self._energy_handle != None):
            self._energy_handle.append_prepare(self._DT_Count)
        
        if (self._VAF_handle != None):
            self._VAF_handle.append_prepare(self._DT_Count)
            
            
        
        if (self._USE_C):
        
            self._constants = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
            
            self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
            self._p1 = loop.SingleAllParticleLoop(self._N,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG)

            self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
            self._p2 = loop.SingleAllParticleLoop(self._N,self._kernel2,{'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG)  
              
            
        self._velocity_verlet_integration()
        if (timer==True):
            end = time.clock()
            print "integrate time taken:", end - start,"s"        
        
        
        
        
    def _velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        self._percent_count = 101
        if (self._plot_handle != None):
            self._percent_int = self._plot_handle.interval
            self._percent_count = percent_int

        self._domain.BCexecute()
        self._state.forces_update()

        for i in range(self._max_it):         
            
            self._velocity_verlet_step()
            
            self._integration_internals(i)
    
        
    
                
    def _velocity_verlet_step(self):
        """
        Perform one step of Velocity Verlet.
        """
        
        if (self._USE_C):
            self._p1.execute()
        else:
            self._V.Dat[...,...]+=0.5*self._dt*self._A.Dat
            self._P.Dat[...,...]+=self._dt*self._V.Dat
        
        #update forces
        self._domain.BCexecute()
        
        self._state.forces_update()
        
        if (self._USE_C):
            self._p2.execute()
        else:
            self._V.Dat[...,...]+= 0.5*self._dt*self._A.Dat
        

    def _integration_internals(self, i):
        
        DTFLAG = ( ((i + 1) % (self._max_it/self._DT_Count) == 0) | (i == (self._max_it-1)) )
        PERCENT = ((100.0*i)/self._max_it)
        
        if ((self._energy_handle != None) & (DTFLAG == True)):
            
            
            self._K.scale(0.0)
            
            if(self._USE_C):
                
                self._pK.execute()
                #self._K.AverageUpdate()
            else:
                for ix in range(self._state.N):
                    self._K[0] += np.sum(self._V[ix,...]*self._V[ix,...])*0.5*self._M[ix]
            
            
            self._energy_handle.U_append(self._state.U.Dat/self._N)
            self._energy_handle.K_append((self._K[0])/self._N)
            self._energy_handle.Q_append((self._state.U[0] + self._K[0])/self._N)
            self._energy_handle.T_append((i+1)*self._dt)
        
        
            
        if ( (self._writexyz == True) & (DTFLAG == True) ):
            self._state.positions.XYZWrite(append=1)
        
           
        #if (DTFLAG==True):
        #        print "Temperature = ",(self._K.Average/self._N)*(2./3.)
        #        self._K.AverageReset()
               
        if ( (self._VAF_handle != None) & (DTFLAG == True) ):    
            self._VAF_handle.evaluate(T=(i+1)*self._dt)
                      
        
        
        if ( (self._plot_handle != None)  & (PERCENT > self._percent_count)):
            
            if (self._plot_handle != None):
                self._plot_handle.draw(self._state.N, self._P, self._state.domain.extent)
            
            self._percent_count += self._percent_int
            print int((100.0*i)/self._max_it),"%", "T=", self._dt*i
                
                
                

################################################################################################################
# Anderson thermostat
################################################################################################################ 

class VelocityVerletAnderson(VelocityVerlet):
    
    def integrate_thermostat(self, dt = None, DT = None, T = None, Temp = 273.15, nu = 1.0, timer=False):
        '''
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double T: End time.
        :arg double Temp: Temperature of heat bath.
        :arg bool timer: display approximate timing information.
        '''
        if (timer==True):
            start = time.clock()
        
        self._Temp = Temp
        self._nu = nu
        
        if (dt != None):
            self._dt = dt
        if (T != None):
            self._T = T
        if (DT != None):
            self._DT = DT
        else:
            self._DT = 50.0*self._dt
            
        self._max_it = int(math.ceil(self._T/self._dt))
        self._DT_Count = int(math.ceil(self._T/self._DT))
        if (self._energy_handle != None):
            self._energy_handle.append_prepare(self._DT_Count)
            
        if (self._VAF_handle != None):
            self._VAF_handle.append_prepare(self._DT_Count)
        
        if (self._USE_C):
            self._constants1 = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
            self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants1)
            self._p1 = loop.SingleAllParticleLoop(self._N,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG)
            
            
            
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
            self._p2_thermostat = loop.SingleAllParticleLoop(self._N,self._kernel2_thermostat,{'V':self._V,'A':self._A, 'M':self._M}, DEBUG = self._DEBUG)  
            
        
        self._velocity_verlet_integration_thermostat()
        
        if (timer==True):
            end = time.clock()
            print "integrate thermostat time taken:", end - start,"s"           
    
    def _velocity_verlet_integration_thermostat(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        
        self._percent_count = 101
        if (self._plot_handle != None):
            self._percent_int = self._plot_handle.interval
            self._percent_count = percent_int

        self._domain.BCexecute()
        self._state.forces_update()

        for i in range(self._max_it):
              
            self._p1.execute()
            
            #update forces
            self._domain.BCexecute()
            self._state.forces_update()
            
            self._p2_thermostat.execute() 
            
            self._integration_internals(i)
            
  
    
    
    
    
    
################################################################################################################
# G(R)
################################################################################################################  
    
class RadialDistributionPeriodicNVE(object):
    '''
    Class to calculate radial distribution function.
    
    :arg state state: State containing particle positions.
    :arg double rmax: Maximum radial distance.
    :arg int rsteps: Resolution to record to, default 100.
    :arg bool DEBUG: Flag to enable debug flags.
    '''
    def __init__(self, state, rmax = None, rsteps = 100, DEBUG = False):
        
        self._count = 0
        self._state = state
        self._extent = self._state.domain.extent
        self._P = self._state.positions
        self._N = self._P.npart
        self._rmax = rmax
        
        if (self._rmax == None):
            self._rmax = 0.5*self._extent.min
        
        
        self._rsteps = rsteps
        
        self._gr = data.ScalarArray(ncomp = self._rsteps, dtype=ctypes.c_int)
        self._gr.scale(0.0)
        
        self._DEBUG = DEBUG
        
        _headers = ['math.h','stdio.h']
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
        
        
        _grkernel = kernel.Kernel('radial_distro_periodic_static',_kernel, _constants, headers = _headers)
        _datdict = {'P':self._P, 'GR':self._gr}
        
        
        self._p = pairloop.DoubleAllParticleLoopOpenMP(N = self._N, kernel = _grkernel, particle_dat_dict = _datdict, DEBUG = self._DEBUG)
        
    def evaluate(self, timer=False):
        '''
        Evaluate the radial distribution function.
        
        :arg bool timer: display approximate timing information.
        '''
        
        assert self._rmax <= 0.5*self._state.domain.extent.min, "Maximum radius too large."
        
        
        if (timer==True):
            start = time.clock()    
        self._p.execute()
        self._count+=1
        if (timer==True):
            end = time.clock()
            print "rdf time taken:", end - start,"s" 
        
    def _scale(self):
        self._r =  np.linspace(0.+0.5*(self._rmax/self._rsteps), self._rmax-0.5*(self._rmax/self._rsteps), num=self._rsteps, endpoint=True)            
        self._grscaled = self._gr.Dat*self._state.domain.volume/((self._N)*(self._N - 1)*2*math.pi*(self._r**2) * (self._rmax/float(self._rsteps))*self._count)
        
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
    
    
    def RawWrite(self, dirname = './output',filename = None, rename_override = False):
        '''
        Function to write Radial Distribution Evaluations to disk.
        
        :arg str dirname: directory to write to, default ./output.
        :arg str filename: Filename to write to, default array name or data.rdf if name unset.
        :arg bool rename_override: Flagging as True will disable autorenaming of output file.
        '''
        
        
        if (filename == None):
            filename = 'data.rdf'
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        
        if (os.path.exists(os.path.join(dirname,filename)) & (rename_override != True)):
            filename=re.sub('.rdf',datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.rdf',filename)
            if (os.path.exists(os.path.join(dirname,filename))):
                filename=re.sub('.rdf',datetime.datetime.now().strftime("_%f") + '.rdf',filename)
                assert os.path.exists(os.path.join(dirname,filename)), "RawWrite Error: No unquie name found."
        
        self._scale()
        f=open(os.path.join(dirname,filename),'w')            
        
        
        f.write('r \t g(r)\n')
        for ix in range(self._gr.ncomp):
            f.write(str(self._r[ix]) + '\t' + str(self._grscaled[ix]) + '\n')
        
        
        f.close()


################################################################################################################
# VAF
################################################################################################################ 

class VelocityAutoCorrelation(object):
    '''
    Method to calculate Velocity Autocorrelation Function.
    
    :arg state state: Input state containing velocities.
    :arg int size: Initial length of VAF array (optional).
    :arg particle.Dat V0: Initial velocity Dat (optional).
    :arg bool DEBUG: Flag to enable debug flags.
    '''
    def __init__(self, state, size = 0, V0 = None, DEBUG = False):
        self._DEBUG = DEBUG
        self._state = state
        self._N = self._state.N
        self._V0 = particle.Dat(self._N, 3, name='V0')
        self._VT = self._V0
        
        self._Ni = data.ScalarArray(val = 1./self._N, dtype = ctypes.c_double)
        
        self._VO_SET = False
        if (V0 != None):
            self.SetV0(V0)
        else:
            self.SetV0(state = self._state)
        
        
        self._VAF = data.ScalarArray(ncomp=size)
        self._VAF_index = data.ScalarArray(val = 0, dtype = ctypes.c_int)
        
        self._T_store = data.ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._T_base = None
        
        _headers = None
        _constants = None
        _kernel_code = '''
        
        
        const double tmp = (V0[0]*VT[0] + V0[1]*VT[1] + V0[2]*VT[2])*Ni[0];
        
        VAF[I[0]] += tmp;
        
        '''
        
        _kernel = kernel.Kernel('VelocityAutocorrelation',_kernel_code, _constants, _headers)
        
        self._datdict = {'VAF':self._VAF, 'V0':self._V0, 'VT':self._VT, 'I':self._VAF_index, 'Ni':self._Ni}
        
        self._loop = loop.SingleAllParticleLoop(N = self._N, kernel = _kernel, particle_dat_dict = self._datdict, DEBUG = self._DEBUG)
        
        
        
        
    def SetV0(self, V0=None, state=None):
        '''
        Set an initial velocity Dat to use as V_0. Requires either a velocity Dat or a state as an argument. V_0 will be set to either the passed velocities or to the velocities in the passed state.        
        
        :arg particle.Dat V0: Velocity Dat.
        :arg state state: State class containing velocities.
        '''
        
        if (V0!=None):
            self._V0.Dat = np.copy(V0.Dat)
            self._V0_SET = True
        if (state!=None):
            self._V0.Dat = np.copy(state.velocities.Dat)
            self._V0_SET = True            
        assert self._V0_SET == True, "No velocities set, check input data."
        
    
    
        
    def evaluate(self, T=None, timer = False):
        '''
        Evaluate VAF using the current velocities held in the state with the velocities in V0.
        
        :arg double T: Time within block of integration.
        :arg bool timer: Flag to time evaluation of VAF.
        '''
        if (timer==True):
            start = time.clock()
        
        
        assert int(self._VAF_index.Dat) < int(self._VAF.ncomp), "VAF store not large enough"
        
        self._Ni.Dat = 1./self._N
        self._datdict['VT'] = self._state.velocities     
        self._loop.execute(self._datdict)
        
        if (T==None):
            self._T_store[self._VAF_index] = 1 + self._T_base
        else:
            
            self._T_store[self._VAF_index.Dat] = T + self._T_base
        
        self._VAF_index.Dat+=1
        
        
        if (timer==True):
            end = time.clock()
            print "VAF time taken:", end - start,"s"         
    
    
        
    def append_prepare(self,size):
        '''
        Function to prepare storage arrays for forthcoming VAF evaluations.
        
        :arg int size: Number of upcoming evaluations.
        '''
        self._VAF.concatenate(size)      
        
        if (self._T_base == None):
            self._T_base = 0.0
        else:
            self._T_base = self._T_store[-1]        
        self._T_store.concatenate(size)
        
    
    def plot(self):
        '''
        Plot array of recorded VAF evaluations.
        '''
        
        if (self._VAF_index > 0):
            plt.ion()
            _fig = plt.figure()
            _ax = _fig.add_subplot(111)
            
            plt.plot(self._T_store.Dat, self._VAF.Dat)
            _ax.set_title('Velocity Autocorrelation Function')
            _ax.set_xlabel('Time')
            _ax.set_ylabel('VAF')
            plt.show()
        else:
            print "Warning: run evaluate() at least once before plotting."





















    
