import math
import state
import data
import pairloop
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kernel
import constant
import ctypes
np.set_printoptions(threshold='nan')

################################################################################################################
# Velocity Verlet Method
################################################################################################################ 

class VelocityVerlet():
    '''
    Class to apply Velocity-Verlet to a given state using a given looping method.
    
    :arg double dt: Time step size, can be specified at integrate call.
    :arg double T: End time, can be specified at integrate call.
    :arg bool USE_C: Flag to use C looping and kernel.
    :arg bool USE_PLOTTING: Flag to plot state at certain progress points.
    :arg bool USE_LOGGING: Flag to log energy at each iteration.
    '''
    
    def __init__(self, dt = 0.0001, T = 0.01, DT = 0.001,state = None, USE_C = True, plot_handle = None, energy_handle = None):
    
        self._dt = dt
        self._DT = DT
        self._T = T
        

        self._state = state
        
        self._domain = self._state.domain()
        self._N = self._state.N()
        self._A = self._state.accelerations()
        self._V = self._state.velocities()
        self._P = self._state.positions()
        self._M = self._state.masses()
        self._K = self._state.K()
        
        
        self._USE_C = USE_C
        self._plot_handle = plot_handle
        self._energy_handle = energy_handle
        
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
        //self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat()
        const double M_tmp = 1/M[0];
        V[0] += dht*A[0]*M_tmp;
        V[1] += dht*A[1]*M_tmp;
        V[2] += dht*A[2]*M_tmp;
        '''
        
        self._K_kernel_code = '''
        K[0]+= (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];
        
        '''      
        self._constants_K = []
        self._K_kernel = kernel.Kernel('K_kernel',self._K_kernel_code,self._constants_K)
        self._pK = pairloop.SingleAllParticleLoop(self._N,self._K_kernel,{'V':self._V,'K':self._K, 'M':self._M},headers = ['stdio.h'])         
        
               
        
    def integrate(self, dt = None, DT = None, T = None):
        '''
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double T: End time.
        '''
        
        
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
        self._energy_handle.append_prepare(self._DT_Count)
        
            
        
        if (self._USE_C):
        
            self._constants = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
            
            
            self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
            self._p1 = pairloop.SingleAllParticleLoop(self._N,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M})

            self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
            self._p2 = pairloop.SingleAllParticleLoop(self._N,self._kernel2,{'V':self._V,'A':self._A, 'M':self._M})  
              
            
        self._velocity_verlet_integration()
        
        
        
    def _velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        

        percent_int = 25
        percent_count = percent_int

        self._domain.boundary_correct(self._P)
        self._state.accelerations_update()

        for i in range(self._max_it):
            
            
            self._velocity_verlet_step()
            
            if ((self._energy_handle != None) & ( ((i + 1) % (self._max_it/self._DT_Count) == 0) | (i == (self._max_it-1)) )):
            
                #self._state.K()._Dat = ( 0.5*np.sum(self._V.Dat()*self._V.Dat()) )
                
                self._K[0] = 0.0
                if(self._USE_C):
                    self._pK.execute()
                else:
                    for ix in range(self._state.N()):
                        self._K += np.sum(self._V[ix,...]*self._V[ix,...])*0.5*self._M[ix]
                
                
                self._energy_handle.U_append(self._state.U().Dat()/self._state.N())
                self._energy_handle.K_append((self._K[0])/self._state.N())
                self._energy_handle.Q_append((self._state.U()[0] + self._K[0])/self._state.N())
                self._energy_handle.T_append((i+1)*self._dt)
            
                
            
            
            if ( ( (self._energy_handle != None) | (self._plot_handle != None) ) & (((100.0*i)/self._max_it) > percent_count)):
                
                if (self._plot_handle != None):
                    self._plot_handle.draw(self._state.N(), self._P, self._state.domain().extent())
                
                percent_count += percent_int
                if (self._energy_handle != None):
                    print int((100.0*i)/self._max_it),"%", "T=", self._dt*i   
    
        
    
                
    def _velocity_verlet_step(self):
        """
        Perform one step of Velocity Verlet.
        """
        
        if (self._USE_C):
            self._p1.execute()
        else:
            self._V.Dat()[...,...]+=0.5*self._dt*self._A.Dat()
            self._P.Dat()[...,...]+=self._dt*self._V.Dat()
        
        #update accelerations
        self._domain.boundary_correct(self._P)
        self._state.accelerations_update()
        
        if (self._USE_C):
            self._p2.execute()
        else:
            self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat()
        

################################################################################################################
# Anderson thermostat
################################################################################################################ 

class VelocityVerletAnderson(VelocityVerlet):
    
    def integrate_thermostat(self, dt = None, DT = None, T = None, Temp = 273.15, nu = 1.0):
        '''
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double T: End time.
        :arg double Temp: Temperature of heat bath.
        '''
        
        self._Temp = Temp
        self._nu = nu
        
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
        self._energy_handle.append_prepare(self._DT_Count)
        
        
        if (self._USE_C):
            self._constants1 = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
            self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants1)
            self._p1 = pairloop.SingleAllParticleLoop(self._N,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M})
            
            
            
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
            
            self._kernel2_thermostat = kernel.Kernel('vv2_thermostat',self._kernel2_thermostat_code,self._constants2_thermostat)
            self._p2_thermostat = pairloop.SingleAllParticleLoop(self._N,self._kernel2_thermostat,{'V':self._V,'A':self._A, 'M':self._M}, headers = ['math.h','stdlib.h','time.h','stdio.h'])  
            
            
            
            
            
            
            
        
        self._velocity_verlet_integration_thermostat()    
    
    def _velocity_verlet_integration_thermostat(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        

        percent_int = 25
        percent_count = percent_int

        self._domain.boundary_correct(self._P)
        self._state.accelerations_update()

        for i in range(self._max_it):
            
            
            self._velocity_verlet_step_thermostat()
            
            if ((self._energy_handle != None) & ( ((i + 1) % (self._max_it/self._DT_Count) == 0) | (i == (self._max_it-1)) )):
            
                #self._state.K()._Dat = ( 0.5*np.sum(self._V.Dat()*self._V.Dat()) )
                
                self._K[0] = 0.0
                if(self._USE_C):
                    self._pK.execute()
                else:
                    for ix in range(self._state.N()):
                        self._K += np.sum(self._V[ix,...]*self._V[ix,...])*0.5*self._M[ix]
                
                
                self._energy_handle.U_append(self._state.U().Dat()/self._state.N())
                self._energy_handle.K_append((self._K[0])/self._state.N())
                self._energy_handle.Q_append((self._state.U()[0] + self._K[0])/self._state.N())
                self._energy_handle.T_append((i+1)*self._dt)
            
                
            
            
            if ( ( (self._energy_handle != None) | (self._plot_handle != None) ) & (((100.0*i)/self._max_it) > percent_count)):
                
                if (self._plot_handle != None):
                    self._plot_handle.draw(self._state.N(), self._P, self._state.domain().extent())
                
                percent_count += percent_int
                if (self._energy_handle != None):
                    print int((100.0*i)/self._max_it),"%", "T=", self._dt*i   
    
        
    
                
    def _velocity_verlet_step_thermostat(self):
        """
        Perform one step of Velocity Verlet.
        """
        
        self._p1.execute()
        
        #update accelerations
        self._domain.boundary_correct(self._P)
        self._state.accelerations_update()
        
        self._p2_thermostat.execute()   
    
    
    
    
    
    
    
    
    
    
    
    
    



    
################################################################################################################
# G(R)
################################################################################################################  
    
class RadialDistributionPeriodicNVE():
    '''
    Class to calculate radial distribution function.
    
    :arg np.array(3,1) positions: Particle positions.
    :arg np.array(3,1) extents: Domain extents.
    :arg double rmax: Maximum radial distance.
    :arg int rsteps: Resolution to record to, default 100.
    :arg np.array(3,1) extent: Domain extents.
    '''
    def __init__(self, state, rmax = 1.0, rsteps = 100):
        
        self._count = 0
        self._state = state
        self._P = self._state.positions()
        self._N = self._P.npart
        self._rmax = rmax
        self._rsteps = rsteps
        self._extent = self._state.domain().extent()
        self._gr = data.ScalarArray(ncomp = self._rsteps)
        
        
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
            /*
            while(abs_md(r20 - r21) > rmaxoverrsteps ){
                r20 = r21;
                r21 -= 0.5*(r21 - (r2/r21) );
            }
            */
            
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
        
        
        _grkernel = kernel.Kernel('radial_distro_periodic_static',_kernel, _constants)
        _datdict = {'P':self._P, 'GR':self._gr}
        _headers = ['math.h']
        
        self._p = pairloop.DoubleAllParticleLoop(N = self._N, kernel = _grkernel, particle_dat_dict = _datdict, headers = _headers)
        
    def evaluate(self):
        self._p.execute()
        self._count+=1
        
    def plot(self):
        if (self._count > 0):
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            r =  np.linspace(0, self._rmax, num=self._rsteps, endpoint=True)
            plt.plot(r,self._state.domain().volume()*self._gr.Dat()/(self._count*(self._N**2)))
            self._ax.set_title('Radial Distribution Function')
            self._ax.set_xlabel('r')
            self._ax.set_ylabel('G(r)')
    
    
    def reset(self):
        self._gr.scale(0.0)
    




























    
