import numpy as np
import ctypes

class Dat():
    '''
    Base class to hold floating point properties of particles, creates N1*N2 array with given initial value.
    
    :arg int N1: First dimension extent.
    :arg int N2: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    
    '''
    def __init__(self, N1 = 1, N2 = 1, initial_value = None, name = None):
        
        self._name = name
        self._type = 'array'
        
        self._N1 = N1
        self._N2 = N2
        
        if (initial_value != None):
            self._Dat = float(initial_value) * np.ones([N1, N2], dtype=ctypes.c_double, order='C')
        else:
            self._Dat = np.zeros([N1, N2], dtype=ctypes.c_double, order='C')
        
    def set_val(self,val):
        self._Dat[...,...] = val
    
     
    def Dat(self):
        '''
        Returns entire data array.
        '''
        return self._Dat
    '''    
    def __getitem__(self,key=None):   
        if (key != None):
            return self._Dat[key]
        else:
            return self._Dat
    '''        
    
    def __getitem__(self,ix):
        return self._Dat[ix] 
            

    def __setitem__(self, ix, val):
        self._Dat[ix] = val      
        
        
    def __str__(self):
        return str(self._Dat)
    
    def __call__(self):
        return self._Dat
    @property    
    def ncomp(self):
        '''
        Return number of components.
        '''
        return self._N2
   
    @property    
    def npart(self):
        '''Return number of particles.'''
        return self._N1
    
    @property    
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
        
        
        
        
        
        
        
        
        



