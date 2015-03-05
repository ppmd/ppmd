import math
import state
import pairloop
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold='nan')

class draw_particles():
    def __init__(self,N,pos,extents):
        print "plotting....."
        plt.ion()
        
        self._N = N
        self._pos = pos
        self._extents = extents

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        self._key=['red','blue']
        plt.show()


    def draw(self):

        plt.cla()
           
        for ix in range(self._N):
            self._ax.scatter(self._pos[ix,0], self._pos[ix,1], self._pos[ix,2],color=self._key[ix%2])
        self._ax.set_xlim([-0.5*self._extents[0],0.5*self._extents[0]])
        self._ax.set_ylim([-0.5*self._extents[1],0.5*self._extents[1]])
        self._ax.set_zlim([-0.5*self._extents[2],0.5*self._extents[2]])
                
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')
        
        plt.draw()
    
class BasicEnergyStore():
    def __init__(self, size = 1):
    
        self._size = size
    
        self._U_store = np.zeros([self._size], dtype=ctypes.c_double, order='C')
        self._K_store = np.zeros([self._size], dtype=ctypes.c_double, order='C')
        self._Q_store = np.zeros([self._size], dtype=ctypes.c_double, order='C')
        self._T_store = np.zeros([self._size], dtype=ctypes.c_double, order='C')
    
        self._U_c = 0
        self._K_c = 0
        self._Q_c = 0
        self._T_c = 0
    
    
    def U_append(self,val):    
        self._U_store[self._U_c] = val
        self._U_c+=1
    def K_append(self,val):    
        self._K_store[self._K_c] = val
        self._K_c+=1        
    def Q_append(self,val):    
        self._Q_store[self._Q_c] = val
        self._Q_c+=1
    def T_append(self,val):    
        self._T_store[self._T_c] = val
        self._T_c+=1            
   
    def plot(self):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(self._T_store,self._Q_store,color='r', linewidth=2)
        ax2.plot(self._T_store,self._U_store,color='g')
        ax2.plot(self._T_store,self._K_store,color='b')
        
        ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kenetic energy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy')
        
        plt.show()    
  

