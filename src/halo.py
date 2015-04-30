import numpy as np
import math
import ctypes
import data

################################################################################################################
# HALO DEFINITIONS
################################################################################################################
             
class HaloCartesian(object):
    """
    Class to contain and control cartesian halo transfers.
    
    """
    def __init__(self):
        self._halos=[]
        
        
    def add_halo(self, halo = None):
        assert halo != None, "Error: No halo specified."
        self._halos.append(halo)
        
    
    
class halo(object):
    """
    Class to contain a halo.
    """
    def __init__(self, local_index = None, ncol = 1, nrow = 1, dtype = ctypes.c_double):
        
        assert local_index != None, "Error: No local index specified."
        
        self._li = local_index #make linear or 3 tuple? leading towards linear.
        self._nc = ncol
        self._nr = nrow
        self._dt = dtype
        
        self._d = np.zeros((self._nc, self._nr), dtype=self._dt, order='C')
    
    def resize(self, ncol = None, nrow = None):
        """
        Resize halo to given dimensions. If a new dimension size is not given, old dimension size will remain. 
        
        :arg int ncol: New number of columns.
        :arg int row: New number of rows.
        """
        resize=False
        if (ncol != None):
            self._nc = ncol
            resize = True
        if (nrow != None):
            self._nr = nrow
            resize = True
        if (resize):
            self._d = np.zeros((self._nc, self._nr), dtype=self._dt, order='C')
        
        
        
    
    
    def __setitem__(self,ix, val):
        self._d[ix] = np.array([val],dtype=self._dt)    
        
    def __str__(self):
        return str(self._d)
        
    def __getitem__(self,ix):
        return self._d[ix]    
    
    @property
    def ctypes_data(self):
        '''Return ctypes-pointer to data.'''
        return self._d.ctypes.data_as(ctypes.POINTER(self._dt))
        
    @property
    def ncol(self):
        '''Return number of columns.'''
        return self._nc
        
    @property
    def nrow(self):
        '''Return number of rwos.'''    
        return self._nr
        
    @property
    def index(self):
        '''Return local index.'''
        return self._li
        
    @property
    def dtype(self):
        ''' Return Dat c data ctype'''
        return self._dt
