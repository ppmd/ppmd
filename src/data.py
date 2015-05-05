import math
import state
import pairloop
import ctypes
import particle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import os
import re
import pickle
import random
np.set_printoptions(threshold='nan')

ctypes_map = {ctypes.c_double:'double', ctypes.c_int:'int', 'float64':'double', 'int32':'int'}


def XYZWrite(dirname = './output', filename='out.xyz', X = None, title='A',sym='A', N_mol = 1, rename_override=False):
    '''
    Function to write particle positions in a xyz format.
    
    :arg str dirname: Directory to write to default ./output.
    :arg str filename: Filename to write to default out.xyz.
    :arg Dat X: Particle dat containing positions.
    :arg str title: title of molecule default ABC. 
    :arg str sym: Atomic symbol for particles, default A.
    :arg int N_mol: Number of atoms per molecule default 1.
    :arg bool rename_override: Flagging as True will disable autorenaming of output file.
    '''
    assert X!=None, "XYZwrite Error: No data."

    if not os.path.exists(dirname):
        os.makedirs(dirname)    
    
    if (os.path.exists(os.path.join(dirname,filename)) & (rename_override != True)):
        filename=re.sub('.xyz',datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.xyz',filename)
        if (os.path.exists(os.path.join(dirname,filename))):
            filename=re.sub('.xyz',datetime.datetime.now().strftime("_%f") + '.xyz',filename)
            assert os.path.exists(os.path.join(dirname,filename)), "XYZWrite Error: No unquie name found."
        
    space=' '
    
    
    f=open(os.path.join(dirname,filename),'w')
    f.write(str(N_mol)+'\n')
    f.write(str(title)+'\n')
    for ix in range(X.npart):
        f.write(str(sym).rjust(3))
        for iy in range(X.ncomp):
            f.write(space+str('%.5f' % X[ix,iy]))
        f.write('\n')
    f.close()
    




class DrawParticles(object):
    '''
    Class to plot N particles with given positions.
    
    :arg int N: Number of particles.
    :arg np.array(N,3) pos: particle positions.
    :arg np.array(3,1) extent:  domain extents.
    
    
    '''
    def __init__(self, interval = 10):
        
        self._interval = interval
        
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
            self._ax.scatter(self._pos.Dat[ix,0], self._pos.Dat[ix,1], self._pos.Dat[ix,2],color=self._key[ix%2])
        self._ax.set_xlim([-0.5*self._extents[0],0.5*self._extents[0]])
        self._ax.set_ylim([-0.5*self._extents[1],0.5*self._extents[1]])
        self._ax.set_zlim([-0.5*self._extents[2],0.5*self._extents[2]])
                
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')
        
        plt.draw()
    
    @property    
    def interval(self):
        return self._interval

################################################################################################################
# Basic Energy Store
################################################################################################################ 
    
class BasicEnergyStore(object):
    '''
    Class to contain recorded values of potential energy U, kenetic energy K, total energy Q and time T.
    
    :arg int size: Required size of each container.
    '''
    def __init__(self, size = 0):
    
        
        
        self._U_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._K_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._Q_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._T_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
    
    
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
        
        
        #Migrate to scalar dats
        self._U_store.concatenate(size)
        self._K_store.concatenate(size)
        self._Q_store.concatenate(size)
        self._T_store.concatenate(size)
        
    
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
        plt.ion()
        fig2 = plt.figure(num=None)
        ax2 = fig2.add_subplot(111)
        ax2.plot(self._T_store.Dat,self._Q_store.Dat,color='r', linewidth=2)
        ax2.plot(self._T_store.Dat,self._U_store.Dat,color='g')
        ax2.plot(self._T_store.Dat,self._K_store.Dat,color='b')
        
        ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kenetic energy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy')
        
        plt.show()    
################################################################################################################
# Scalar array.
################################################################################################################ 
class ScalarArray(object):
    '''
    Base class to hold a single floating point property.
    
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    
    '''
    def __init__(self, initial_value = None, name = None, ncomp = 1, val = None, dtype=ctypes.c_double):
        '''
        Creates scalar with given initial value.
        '''
        
        self._dtype = dtype
        
        if (name != None):
            self._name = name
        self._N1 = ncomp
        
        if (initial_value != None):
            if (type(initial_value) is np.ndarray):
                self._Dat = np.array(initial_value, dtype=self._dtype, order='C')
                self._N1 = initial_value.shape[0]
            elif (type(initial_value) == list):
                self._Dat = np.array(np.array(initial_value), dtype=self._dtype, order='C')
                self._N1 = len(initial_value)
            else:
                self._Dat = float(initial_value) * np.ones([self._N1], dtype=self._dtype, order='C')
        elif (val == None):
            self._Dat = np.zeros([self._N1], dtype=self._dtype, order='C')
        elif (val != None):
            self._Dat = np.array([val], dtype=self._dtype, order='C')
        
        self._A = False
        
    def concatenate(self, size):
        '''
        Increase length of scalar array object.
        
        :arg int size: Number of new elements.
        '''
        self._Dat = np.concatenate((self._Dat, np.zeros(size, dtype=self._dtype, order='C')))
        self._N1 += size
        if (self._A == True):   
            self._Aarray = np.concatenate((self._Aarray, np.zeros(size, dtype=self._dtype, order='C')))
            self._Aarray = 0.*self._Aarray
        
        
            
    @property
    def Dat(self):
        '''
        Returns stored data as numpy array.
        '''
        return self._Dat
    
    @Dat.setter
    def Dat(self, val):
        self._Dat = np.array([val],dtype=self._dtype)     
    
    @property
    def test(self):
        return self._Dat[0]
        
    def __getitem__(self,ix):
        return self._Dat[ix]
        
    def scale(self,val):
        '''
        Scale data array by value val.
        
        :arg double val: Coefficient to scale all elements by.
        '''
        self._Dat = np.array([val],dtype=self._dtype) * self._Dat
        
    
    def __setitem__(self,ix, val):
        self._Dat[ix] = np.array([val],dtype=self._dtype)
        
        
        if (self._A == True):
            self._Aarray[ix] = np.array([val],dtype=self._dtype)
            self._Alength += 1
            
          
    def __str__(self):
        return str(self._Dat)
    
    def __call__(self):
        return self._Dat
    
    @property      
    def ctypes_data(self):
        '''Return ctypes-pointer to data.'''
        return self._Dat.ctypes.data_as(ctypes.POINTER(self._dtype))
        
    @property
    def dtype(self):
        ''' Return Dat c data ctype'''
        return self._dtype
    
    @property 
    def ctypes_value(self):
        '''Return first value in correct type.'''
        return (self._dtype)(self._Dat[0])
    
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
        
    @property
    def min(self):
        '''Return minimum'''
        return self._Dat.min()
        
    @property
    def max(self):
        '''Return maximum'''
        return self._Dat.max()
    @property
    def mean(self):
        '''Return mean'''
        return self._Dat.mean()      
        
    @property
    def name(self):
        return "ScalarArray"
    
    
        
    def DatWrite(self, dirname = './output',filename = None, rename_override = False):
        '''
        Function to write ScalarArray objects to disk.
        
        :arg str dirname: directory to write to, default ./output.
        :arg str filename: Filename to write to, default array name or data.SArray if name unset.
        :arg bool rename_override: Flagging as True will disable autorenaming of output file.
        '''
        
        
        if (self._name!=None and filename == None):
            filename = str(self._name)+'.SArray'
        if (filename == None):
            filename = 'data.SArray'
          
            
            
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        
        if (os.path.exists(os.path.join(dirname,filename)) & (rename_override != True)):
            filename=re.sub('.SArray',datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.SArray',filename)
            if (os.path.exists(os.path.join(dirname,filename))):
                filename=re.sub('.SArray',datetime.datetime.now().strftime("_%f") + '.SArray',filename)
                assert os.path.exists(os.path.join(dirname,filename)), "DatWrite Error: No unquie name found."
        
        
        f=open(os.path.join(dirname,filename),'w')            
        pickle.dump(self._Dat, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    
    
       
    def DatRead(self, dirname = './output', filename=None):
        '''
        Function to read Dat objects from disk.
        
        :arg str dirname: directory to read from, default ./output.
        :arg str filename: filename to read from.
        '''
        
        assert os.path.exists(dirname), "Read directory not found"         
        assert filename!=None, "DatRead Error: No filename given."
        
            
        f=open(os.path.join(dirname,filename),'r')
        self=pickle.load(f)
        f.close()         
        

    def AverageReset(self):
        '''Resest and initialises averaging.'''
        if (self._A == False):
            
            
            self._Aarray = np.zeros([self._N1], dtype=self._dtype, order='C')
            self._Alength = 0.0
            self._A = True
        else:
            self._Aarray = 0.*self._Aarray
            self._Alength = 0.0
            
    @property 
    def Average(self):
        '''Returns averages of recorded values since AverageReset was called.'''
        #assert self._A == True, "Run AverageReset to initialise or reset averaging"
        if (self._A == True):
            
            return self._Aarray/self._Alength

    def AverageStop(self, clean=False):
        '''
        Stops averaging values.
        
        :arg bool clean: Flag to free memory allocated to averaging, default False.
        '''
        if (self._A == True):
            self._A = False
            if (clean==True):
                del self._A
    
    def AverageUpdate(self):
        '''Copy values from Dat into averaging array'''
        if (self._A == True):
            self._Aarray += self._Dat
            self._Alength += 1
        else:
            self.AverageReset()
            self._Aarray += self._Dat
            self._Alength += 1  
                       
################################################################################################################
# Pointer array.
################################################################################################################ 

class PointerArray(object):
    '''
    Class to store arrays of pointers.
    
    :arg int length: Length of array.
    :arg ctypes.dtype dtype: pointer data type.
    '''
    def __init__(self, length, dtype):
        self._length = length
        self._dtype = dtype
        self._Dat = (ctypes.POINTER(self._dtype)*self._length)()
    
    @property
    def dtype(self):
        '''Returns data type.'''
        return self._dtype
        
    @property      
    def ctypes_data(self):
        '''Returns pointer to start of array.'''
        return self._Dat      























        
        
