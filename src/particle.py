import numpy as np
import ctypes
import datetime
import os
import re
import pickle
import random
import access



class Dat(object):
    '''
    Base class to hold floating point properties of particles, creates N1*N2 array with given initial value.
    
    :arg int N1: First dimension extent.
    :arg int N2: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    
    '''
    def __init__(self, N1 = 1, N2 = 1, initial_value = None, name = None, dtype = ctypes.c_double, max_size = 200000):
        
        self._name = name
        
        self._dtype = dtype
        
        self._N1 = N1
        self._N2 = N2
        self._max_size = max_size
        
        if (initial_value != None):
            if (type(initial_value) is np.ndarray):
                self._Dat = np.array(initial_value, dtype=self._dtype, order='C')
                self._N1 = initial_value.shape[0]
                self._N2 = initial_value.shape[1]   
                self._max_size = initial_value.shape[0]          
            else:
                self._Dat = float(initial_value) * np.ones([N1, N2], dtype=self._dtype, order='C')
                self._max_size = N1
        else:
            self._Dat = np.zeros([max_size, N2], dtype=self._dtype, order='C')
        
        self._XYZFile_exists = False
        
        self._halo_start = self._N1
        
        '''Number of halo particles'''
        
        self._NH = self._halo_start - self._N1
        
        
        
    def set_val(self,val):
        self._Dat[...,...] = val
    
    @property 
    def Dat(self):
        '''
        Returns entire data array.
        '''
        return self._Dat
            
    @Dat.setter
    def Dat(self, val):
        self._Dat = np.array([val],dtype=self._dtype)
    
    
    def __getitem__(self,ix):
        return self._Dat[ix] 
            
    
    def __setitem__(self, ix, val):
        self._Dat[ix] = val
        '''
        if (type(ix)==int):
            tmp=ix
        elif (type(ix)==tuple):
            tmp=ix[0]
        
        
        if (tmp > self._N1):
            self._halo_start = tmp+1
        '''    
        
        
    def __str__(self):
        return str(self._Dat)
    
    
    def __call__(self, access=access.RW, halo=False):
        
                
        
        
        return self
       
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
        
    @npart.setter
    def npart(self,val):
        '''Set number of particles.'''
        self._N1 = val
        
        
        
    @property
    def npart_halo(self):
        '''Return number of particles in halo region'''
        return self._NH    
        
    
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
    
    @property
    def halo_start(self):
        return self._halo_start
    
    def halo_start_shift(self, shift):
        self._halo_start+=shift
        self._NH = self._halo_start - self._N1
        
    def halo_start_set(self,index):
    
    
        if index < self._N1:
            if (index >= 0):
                self._N1 = index
                self._halo_start = index
                
        else:
            self.halo_start_reset()
            
        self._NH = 0
        
        
    def halo_start_reset(self):
        self._halo_start = self._N1
        self._NH = 0
    
    
    
    
    
        
    def resize(self, N):
        '''
        Resize particle dat to be at least a certain size, does not resize if already large enough.
        
        :arg int N: New minimum size.
        '''
        
        if (N>self._max_size):
            self._max_size = N+(N-self._max_size)*10
            self._Dat.resize([N, self._N2], dtype=self._dtype, order='C' )
        #self._N1 = N        
        
        
        
        
        
    
        
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
        
    def XYZWrite(self, dirname = './output', filename='out.xyz', title=None,sym=None, rename_override=False, append = 0):
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
        
        if (append == 0):
            self._XYZFile_exists = False
            self._XYZfilename = filename
            
        if ((append > 0) & (self._XYZFile_exists == False)):
            self._XYZfilename = filename
        
        
        
        
        if (title==None):
            title = 'AA'
        if (sym == None):
            sym = 'A'
            
        

        if not os.path.exists(dirname):
            os.makedirs(dirname)    
        
        
        
        if (os.path.exists(os.path.join(dirname,self._XYZfilename)) & (rename_override == False) & (self._XYZFile_exists == False)):
            self._XYZfilename=re.sub('.xyz',datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.xyz',self._XYZfilename)
            if (os.path.exists(os.path.join(dirname,self._XYZfilename))):
                self._XYZfilename=re.sub('.xyz',datetime.datetime.now().strftime("_%f") + '.xyz',self._XYZfilename)
                assert os.path.exists(os.path.join(dirname,self._XYZfilename)), "XYZWrite Error: No unquie name found."
        self._XYZFile_exists = True
        
            
        space=' '
        
        if (append == 0):
            f=open(os.path.join(dirname,self._XYZfilename),'w')
        if (append > 0):
            f=open(os.path.join(dirname,self._XYZfilename),'a')
        
        f.write(str(self._N1)+'\n')
        f.write(str(title)+'\n')
        s=''
        for ix in range(self._N1):
            
            #f.write(str(sym).rjust(3))
            s+=str(sym).rjust(3)
            
            for iy in range(self._N2):
                #f.write(space+str('%.5f' % self._Dat[ix,iy]))
                s+=space+str('%.5f' % self._Dat[ix,iy])
            #f.write('\n')
            s+='\n'
        f.write(s)
        f.close()        
        
        
        
        
        
        
        
class TypedDat(Dat):
    '''
    Base class to hold floating point properties of particles based on particle type, creates N1*N2 array with given initial value.
    
    :arg int N1: First dimension extent.
    :arg int N2: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    
    '''
    def __init__(self, N1 = 1, N2 = 1, initial_value = None, name = None, dtype = ctypes.c_double, max_size = 100000):
        
        self._name = name
        
        self._dtype = dtype
        
        self._N1 = N1
        self._N2 = N2
        self._max_size = max_size
        
        if (initial_value != None):
            if (type(initial_value) is np.ndarray):
                self._Dat = np.array(initial_value, dtype=self._dtype, order='C')
                self._N1 = initial_value.shape[0]
                self._N2 = initial_value.shape[1]   
                self._max_size = initial_value.shape[0]          
            else:
                self._Dat = float(initial_value) * np.ones([N1, N2], dtype=self._dtype, order='C')
                self._max_size = N1
        else:
            self._Dat = np.zeros([max_size, N2], dtype=self._dtype, order='C')
        
        self._XYZFile_exists = False
        
        self._halo_start = self._N1
        
        '''Number of halo particles'''
        
        self._NH = self._halo_start - self._N1        
        
        
        



