import math
import state
import pairloop
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
np.set_printoptions(threshold='nan')

class draw_particles():
    '''
    Class to plot N particles with given positions.
    
    :arg int N: Number of particles.
    :arg np.array(N,3) pos: particle positions.
    :arg np.array(3,1) extent:  domain extents.
    
    
    '''
    def __init__(self):
        print "plotting....."
        plt.ion()
        

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        self._key=['red','blue']
        plt.show()
        
    def draw(self,N,pos,extents):
        '''
        Update current plot, use for real time plotting.
        '''
        
        self._N = N
        self._pos = pos
        self._extents = extents
        
        plt.cla()
           
        for ix in range(self._N):
            self._ax.scatter(self._pos.Dat()[ix,0], self._pos.Dat()[ix,1], self._pos.Dat()[ix,2],color=self._key[ix%2])
        self._ax.set_xlim([-0.5*self._extents[0],0.5*self._extents[0]])
        self._ax.set_ylim([-0.5*self._extents[1],0.5*self._extents[1]])
        self._ax.set_zlim([-0.5*self._extents[2],0.5*self._extents[2]])
                
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')
        
        plt.draw()

################################################################################################################
# Basic Energy Store
################################################################################################################ 
    
class BasicEnergyStore():
    '''
    Class to contain recorded values of potential energy U, kenetic energy K, total energy Q and time T.
    
    :arg int size: Required size of each container.
    '''
    def __init__(self, size = 0):
    
        self._U_store = np.zeros([size], dtype=ctypes.c_double, order='C')
        self._K_store = np.zeros([size], dtype=ctypes.c_double, order='C')
        self._Q_store = np.zeros([size], dtype=ctypes.c_double, order='C')
        self._T_store = np.zeros([size], dtype=ctypes.c_double, order='C')
    
        self._U_c = 0
        self._K_c = 0
        self._Q_c = 0
        self._T_c = 0
        self._T_base = None
        
    def append_prepare(self,size):
        
        if (self._T_base == None):
            self._T_base = 0.0
        else:
            self._T_base = self._T_store[-1]
        
        
        
        self._U_store = np.concatenate((self._U_store, np.zeros(size, dtype=ctypes.c_double, order='C')))
        self._K_store = np.concatenate((self._K_store, np.zeros(size, dtype=ctypes.c_double, order='C')))
        self._Q_store = np.concatenate((self._Q_store, np.zeros(size, dtype=ctypes.c_double, order='C')))
        self._T_store = np.concatenate((self._T_store, np.zeros(size, dtype=ctypes.c_double, order='C')))
        
    
    def U_append(self,val):
        '''
        Append a value to potential energy.
        
        :arg double val: value to append
        '''
        
        self._U_store[self._U_c] = val
        self._U_c+=1
    def K_append(self,val): 
        '''
        Append a value to kenetic energy.
        
        :arg double val: value to append
        '''       
        self._K_store[self._K_c] = val
        self._K_c+=1        
    def Q_append(self,val): 
        '''
        Append a value to total energy.
        
        :arg double val: value to append
        '''       
        self._Q_store[self._Q_c] = val
        self._Q_c+=1
    def T_append(self,val):
        '''
        Append a value to time store.
        
        :arg double val: value to append
        '''       
        self._T_store[self._T_c] = val + self._T_base
        self._T_c+=1            
   
    def plot(self):
        '''
        Plot recorded energies against time.
        '''
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(self._T_store,self._Q_store,color='r', linewidth=2)
        ax2.plot(self._T_store,self._U_store,color='g')
        ax2.plot(self._T_store,self._K_store,color='b')
        
        ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kenetic energy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy')
        
        plt.show()    
################################################################################################################
# Scalar array.
################################################################################################################ 
class ScalarArray():
    '''
    Base class to hold a single floating point property.
    
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    
    '''
    def __init__(self, initial_value = None, name = None, ncomp = 1, val = None):
        '''
        Creates scalar with given initial value.
        '''
        
        self._type = 'scalar'
        if (name != None):
            self._name = name
        self._N1 = ncomp
        
        if (initial_value != None):
            self._Dat = float(initial_value) * np.ones([self._N1], dtype=ctypes.c_double, order='C')
        if (val == None):
            self._Dat = np.zeros([self._N1], dtype=ctypes.c_double, order='C')
        if (val != None):
            self._Dat = val.astype(dtype=ctypes.c_double, order='C')
        
        
        
        
        
    
    def Dat(self):
        '''
        Returns stored data as numpy array.
        '''
        return self._Dat
        
    def __getitem__(self,ix):
        return self._Dat[ix]
        
    def scale(self,val):
        '''
        Scale data array by value val.
        
        :arg double val: Coefficient to scale all elements by.
        '''
        self._Dat = val * self._Dat
        
    
    def __setitem__(self,ix, val):
        self._Dat[ix] = val
          
    def __str__(self):
        return str(self._Dat)
    
    def __call__(self):
        return self._Dat
    
          
    def ctypes_data(self):
        '''Return ctypes-pointer to data.'''
        return self._Dat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))        
    @property     
    def dattype(self):
        '''
        Returns type of particle dat.
        '''    
        return self._type            
        
    @property
    def name(self):
        '''
        Returns name of particle dat.
        '''    
        return self._name    
    
    @property
    def ncomp(self):
        '''
        Return number of components.
        '''   
        return self._N1
