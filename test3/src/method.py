import math
import state
import data
import pairloop
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold='nan')

class VelocityVerlet():
    """
    Class to apply Velocity-Verlet to a given state using a given looping method.
    """
    
    def __init__(self, dt = 0.0001, T = 0.01, looping_method_accel = None, state = None):
        self._dt = dt
        self._T = T
        
        
        self._state = state
        
        
        if (looping_method_accel == None):
            self._looping_method_accel = pairloop.PairLoopRapaport(self._state)
        else:
            self._looping_method_accel = looping_method_accel
        
        
        
        self._A = self._state.accelerations().Dat()
        self._V = self._state.velocities().Dat()
        self._P = self._state.positions().Dat()
    
        ''' Drawing particles initialisation'''
        self._pos_draw = data.draw_particles(self._state.N(), self._P, self._state.domain().extent())
    
        
    
    def integrate(self, dt = None, T = None):
        if (dt != None):
            self._dt = dt
        if (T != None):
            self._T = T
        self._max_it = int(math.ceil(self._T/self._dt))
        
        self._E_store = data.BasicEnergyStore(self._max_it)
        
        self.velocity_verlet_integration()
        
        return self._E_store
        
        
    def velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    

        

        percent_int = 10
        percent_count = percent_int

        self._looping_method_accel.update()

        for i in range(self._max_it):
            
            
            self.velocity_verlet_step()
            
            if (i > -1):
                self._state.K_set( 0.5*np.sum(self._V*self._V) )
                
                
                self._E_store.U_append(self._state.U()/self._state.N())
                self._E_store.K_append(( 0.5*np.sum(self._V*self._V) )/self._state.N())
                self._E_store.Q_append((self._state.U() + self._state.K())/self._state.N())
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
        
        self._V+=0.5*self._dt*self._A
        self._P+=self._dt*self._V
        
        #update accelerations
        self._looping_method_accel.update()
        
        self._V+= 0.5*self._dt*self._A
    
    
 
    
    
    
    
    
    
    
    
