import math
import state
import data
import pairloop
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kernel
import constant
np.set_printoptions(threshold='nan')

class VelocityVerlet():
    '''
    Class to apply Velocity-Verlet to a given state using a given looping method.
    
    :arg double dt: Time step size, can be specified at integrate call.
    :arg double T: End time, can be specified at integrate call.
    :arg bool USE_C: Flag to use C looping and kernel.
    :arg bool USE_PLOTTING: Flag to plot state at certain progress points.
    :arg bool USE_LOGGING: Flag to log energy at each iteration.
    '''
    
    def __init__(self, dt = 0.0001, T = 0.01, state = None, USE_C = True, USE_PLOTTING = True, USE_LOGGING = True):
    
        self._dt = dt
        self._T = T
        

        self._state = state
        
        self._domain = self._state.domain()
        self._N = self._state.N()
        self._A = self._state.accelerations()
        self._V = self._state.velocities()
        self._P = self._state.positions()
        self._M = self._state.masses()
        self._K = self._state.K()
        
        
    
        ''' Drawing particles initialisation'''
        self._pos_draw = data.draw_particles(self._state.N(), self._P, self._state.domain().extent())
        
        '''Updates step broken into two parts'''
        self._USE_C = USE_C
        self._USE_PLOTTING = USE_PLOTTING
        self._USE_LOGGING = USE_LOGGING
        
    def integrate(self, dt = None, T = None):
        '''
        Integrate state forward in time.
        
        :arg double dt: Time step size.
        :arg double T: End time.
        '''
        if (dt != None):
            self._dt = dt
        if (T != None):
            self._T = T
        self._max_it = int(math.ceil(self._T/self._dt))
        
        self._E_store = data.BasicEnergyStore(self._max_it)
        
        if (self._USE_C):
        
            self._constants = [constant.Constant('dt',self._dt), constant.Constant('dht',0.5*self._dt),]
            
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
            self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
            self._p1 = pairloop.SingleAllParticleLoop(self._N,self._kernel1,{'P':self._P,'V':self._V,'A':self._A, 'M':self._M})
            
            self._kernel2_code = '''
            //self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat()
            const double M_tmp = 1/M[0];
            V[0] += dht*A[0]*M_tmp;
            V[1] += dht*A[1]*M_tmp;
            V[2] += dht*A[2]*M_tmp;
            '''
            
            self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
            self._p2 = pairloop.SingleAllParticleLoop(self._N,self._kernel2,{'V':self._V,'A':self._A, 'M':self._M})  
            
            self._K_kernel_code = '''
            K[0]+= (V[0]*V[0] + V[1]*V[1] + V[2]*V[2])*0.5*M[0];
            
            '''      
            self._constants_K = []
            self._K_kernel = kernel.Kernel('K_kernel',self._K_kernel_code,self._constants_K)
            self._pK = pairloop.SingleAllParticleLoop(self._N,self._K_kernel,{'V':self._V,'K':self._K, 'M':self._M},headers = ['stdio.h']) 
            
            
            
        
        self._velocity_verlet_integration()
        
        return self._E_store
        
        
    def _velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        

        percent_int = 10
        percent_count = percent_int

        self._domain.boundary_correct(self._P)
        self._state.accelerations_update()

        for i in range(self._max_it):
            
            
            self._velocity_verlet_step()
            
            if ((self._USE_LOGGING) & (i > -1)):
            
                #self._state.K()._Dat = ( 0.5*np.sum(self._V.Dat()*self._V.Dat()) )
                
                self._K[0] = 0.0
                if(self._USE_C):
                    self._pK.execute()
                else:
                    for ix in range(self._state.N()):
                        self._K += np.sum(self._V[ix,...]*self._V[ix,...])*0.5*self._M[ix]
                
                
                self._E_store.U_append(self._state.U().Dat()/self._state.N())
                self._E_store.K_append((self._K[0])/self._state.N())
                self._E_store.Q_append((self._state.U()[0] + self._K[0])/self._state.N())
                self._E_store.T_append((i+1)*self._dt)
            
                
            
            
            if ( ( self._USE_LOGGING | self._USE_PLOTTING ) & (((100.0*i)/self._max_it) > percent_count)):
                
                if (self._USE_PLOTTING):
                    self._pos_draw.draw()
                
                percent_count += percent_int
                if (self._USE_LOGGING):
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
        
    
 
    
    
    
    
    
    
    
    
