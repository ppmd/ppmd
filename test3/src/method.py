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
    """
    Class to apply Velocity-Verlet to a given state using a given looping method.
    """
    
    def __init__(self, dt = 0.0001, T = 0.01, looping_method_accel = None, state = None, USE_C = True):
        self._dt = dt
        self._T = T
        
        
        
        self._state = state
        
        
        if (looping_method_accel == None):
            self._looping_method_accel = pairloop.PairLoopRapaport(self._state)
        else:
            self._looping_method_accel = looping_method_accel
        
        
        
        self._A = self._state.accelerations()
        self._V = self._state.velocities()
        self._P = self._state.positions()
        
        
    
        ''' Drawing particles initialisation'''
        self._pos_draw = data.draw_particles(self._state.N(), self._P, self._state.domain().extent())
        
        '''Updates step broken into two parts'''
        self._USE_C = USE_C
        
        
        
        
    
        
    
    def integrate(self, dt = None, T = None):
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
            V[0] += dht*A[0];
            V[1] += dht*A[1];
            V[2] += dht*A[2];
            P[0] += dt*V[0];
            P[1] += dt*V[1];
            P[2] += dt*V[2];
            '''
            self._kernel1 = kernel.Kernel('vv1',self._kernel1_code,self._constants)
            self._p1 = pairloop.SingleAllParticleLoop(self._kernel1,{'P':self._P,'V':self._V,'A':self._A})
            
            self._kernel2_code = '''
            //self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat()
            V[0] += dht*A[0];
            V[1] += dht*A[1];
            V[2] += dht*A[2];
            '''
            
            self._kernel2 = kernel.Kernel('vv2',self._kernel2_code,self._constants)
            self._p2 = pairloop.SingleAllParticleLoop(self._kernel2,{'V':self._V,'A':self._A})        
        
        
        self.velocity_verlet_integration()
        
        
        
        return self._E_store
        
        
    def velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        

        percent_int = 100
        percent_count = percent_int

        self._looping_method_accel.execute()

        for i in range(self._max_it):
            
            
            self.velocity_verlet_step()
            
            if (i > -1):
                self._state.K()._Dat = ( 0.5*np.sum(self._V.Dat()*self._V.Dat()) )
                
                
                self._E_store.U_append(self._state.U().Dat()/self._state.N())
                self._E_store.K_append(( 0.5*np.sum(self._V.Dat()*self._V.Dat()) )/self._state.N())
                self._E_store.Q_append((self._state.U().Dat() + self._state.K().Dat())/self._state.N())
                self._E_store.T_append((i+1)*self._dt)
            
                
            #print i, self.positions()[1,0] - self.positions()[0,0]
            
            if ( ((100.0*i)/self._max_it) > percent_count):
                
                self._pos_draw.draw()
                
                percent_count += percent_int
                print int((100.0*i)/self._max_it),"%", "T=", self._dt*i   
    
        
    
                
    def velocity_verlet_step(self):
        """
        Perform one step of Velocity Verlet.
        """
        
        if (self._USE_C):
            self._p1.execute()
        else:
            self._V.Dat()[...,...]+=0.5*self._dt*self._A.Dat()
            self._P.Dat()[...,...]+=self._dt*self._V.Dat()
        
        #update accelerations
        
        self._looping_method_accel.execute()
        
        
        if (self._USE_C):
            self._p2.execute()
        else:
            self._V.Dat()[...,...]+= 0.5*self._dt*self._A.Dat()
        
    
 
    
    
    
    
    
    
    
    
