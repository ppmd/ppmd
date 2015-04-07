import numpy as np
import ctypes
import datetime
import os
import re
import pickle
import random

class Dat(object):
    '''
    Base class to hold floating point properties of particles, creates N1*N2 array with given initial value.
    
    :arg int N1: First dimension extent.
    :arg int N2: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    
    '''
    def __init__(self, N1 = 1, N2 = 1, initial_value = None, name = None):
        
        self._name = name
        
        self._dtype = ctypes.c_double
        
        self._N1 = N1
        self._N2 = N2
        
        if (initial_value != None):
            self._Dat = float(initial_value) * np.ones([N1, N2], dtype=self._dtype, order='C')
        else:
            self._Dat = np.zeros([N1, N2], dtype=self._dtype, order='C')
        
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
    def dtype(self):
        ''' Return Dat c data ctype'''
        return self._dtype
        
              
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
    
        
    def DatWrite(self, dirname = './output',filename = None, rename_override = False):
        '''
        Function to write Dat objects to disk.
        
        :arg str dirname: directory to write to, default ./output.
        :arg str filename: Filename to write to, default dat name or data.Dat if name unset.
        :arg bool rename_override: Flagging as True will disable autorenaming of output file.
        '''
        
        
        if (self._name!=None and filename == None):
            filename = str(self._name)+'.Dat'
        if (filename == None):
            filename = 'data.Dat'
          
            
            
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        
        if (os.path.exists(os.path.join(dirname,filename)) & (rename_override != True)):
            filename=re.sub('.Dat',datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.Dat',filename)
            if (os.path.exists(os.path.join(dirname,filename))):
                filename=re.sub('.Dat',datetime.datetime.now().strftime("_%f") + '.Dat',filename)
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
        
        
        
        
        
        
        



