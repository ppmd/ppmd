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
from mpi4py import MPI
np.set_printoptions(threshold='nan')

ctypes_map = {ctypes.c_double:'double', ctypes.c_int:'int', 'float64':'double', 'int32':'int','doublepointerpointer':'double **'}
mpi_map = {ctypes.c_double:MPI.DOUBLE, ctypes.c_int:MPI.INT}

################################################################################################################
# MDMPI
################################################################################################################

class MDMPI(object):
    '''
    Class to store a MPI communicator such that it can be used everywhere (bottom level of hierarchy).
    
    '''
    
    def __init__(self):
        self._COMM = None
        
    @property
    def comm(self):
        '''
        Return the current communicator.
        '''
        return self._COMM
    
    @comm.setter
    def comm(self, new_comm = None):
        '''
        Set the current communicator.
        '''    
        assert new_comm != None, "MDMPI error: no new communicator assigned."
        self._COMM = new_comm
        
    def __call__(self):
        '''
        Return the current communicator.
        '''
        return self._COMM        
        
    @property
    def rank(self):
        '''
        Return the current rank.
        '''
        if (self._COMM != None):
            return self._COMM.Get_rank()
        else:
            return 0
    
    @property
    def nproc(self):
        '''
        Return the current size.
        '''
        if (self._COMM != None):  
            return self._COMM.Get_size()
        else:
            return 1
    
    @property
    def top(self):
        '''
        Return the current topology.
        '''
        if (self._COMM != None):
            return self._COMM.Get_topo()[2][::-1]
        else:
            return (0,0,0)
    
    @property
    def dims(self):
        '''
        Return the current dimensions.
        '''    
        if (self._COMM != None):
            return self._COMM.Get_topo()[0][::-1]
        else:
            return (1,1,1)
    
    def barrier(self):
        '''
        alias to comm barrier method.
        '''
        if (self._COMM != None):
            self._COMM.Barrier()
    
    
################################################################################################################
# XYZWrite
################################################################################################################


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
    

################################################################################################################
# DrawParticles
################################################################################################################


class DrawParticles(object):
    '''
    Class to plot N particles with given positions.
    
    :arg int N: Number of particles.
    :arg np.array(N,3) pos: particle positions.
    :arg np.array(3,1) extent:  domain extents.
    
    
    '''
    def __init__(self, interval = 10, MPI_handle = MDMPI()):
        
        
        self._interval = interval
        self._Mh = MPI_handle
        
        self._Dat = None
        self._gids = None
        
        if (self._Mh.rank == 0 or self._Mh == None):
            plt.ion()
            
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            
            self._key=['red','blue']
            plt.show()
        
        
    def draw(self, state):
        '''
        Update current plot, use for real time plotting.
        '''
        self._N = state.N()
        self._NT = state.NT()
        self._extents = state.domain.extent
        
        
        
        '''Case where all particles are local''' 
        if self._Mh == None:
            self._pos = state.positions
            self._gid = state.global_ids
            
        else:
            '''Need an mpi handle if not all particles are local'''
            assert self._Mh != None, "Error: Not all particles are local but MPI_handle = None."
            
            '''Allocate if needed'''
            if self._Dat == None:
                self._Dat = particle.Dat(self._NT, 3)
            else:
                self._Dat.resize(self._NT)
            
            if self._gids == None:
                self._gids = ScalarArray(ncomp = self._NT, dtype=ctypes.c_int)
            else:
                self._gids.resize(self._NT)
            
            
            
            _MS=MPI.Status()
            
            
            if self._Mh.rank == 0:
                
                '''Copy the local data.'''
                self._Dat.Dat[0:self._N:,::] = state.positions.Dat[0:self._N:,::]
                self._gids[0:self._N:] = state.global_ids[0:self._N:]
                
                _i = self._N #starting point pos
                _ig = self._N #starting point gids
                
                for ix in range(1,self._Mh.nproc):
                    self._Mh.comm.Recv(self._Dat.Dat[_i::,::], ix, ix, _MS)
                    _i+=_MS.Get_count( mpi_map[self._Dat.dtype])/3
                    
                    self._Mh.comm.Recv(self._gids.Dat[_ig::], ix, ix, _MS)
                    _ig+=_MS.Get_count( mpi_map[self._gids.dtype])        
                    
                self._pos = self._Dat
                self._gid = self._gids
            else:
                
                self._Mh.comm.Send(state.positions.Dat[0:self._N:,::], 0, self._Mh.rank)
                self._Mh.comm.Send(state.global_ids.Dat[0:self._N:], 0, self._Mh.rank)
        
        if (self._Mh.rank == 0 ):
            
            
            
            
            plt.cla()
            plt.ion()
            for ix in range(self._pos.npart):
                self._ax.scatter(self._pos.Dat[ix,0], self._pos.Dat[ix,1], self._pos.Dat[ix,2],color=self._key[self._gid[ix]%2])
            self._ax.set_xlim([-0.5*self._extents[0],0.5*self._extents[0]])
            self._ax.set_ylim([-0.5*self._extents[1],0.5*self._extents[1]])
            self._ax.set_zlim([-0.5*self._extents[2],0.5*self._extents[2]])
                    
            self._ax.set_xlabel('x')
            self._ax.set_ylabel('y')
            self._ax.set_zlabel('z')
            
            plt.draw()            
        
        self._Mh.barrier()
            
        
    
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
    def __init__(self, size = 0, MPI_handle = MDMPI()):
    
        
        self._U_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._K_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._Q_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        self._T_store = ScalarArray(initial_value = 0.0, ncomp = size, dtype=ctypes.c_double)
        
        self._U_c = 0
        self._K_c = 0
        self._Q_c = 0
        self._T_c = 0
        self._T_base = None
        
        self._Mh = MPI_handle
                
    def append_prepare(self,size):
        
        self._size = size
        
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
        
        if self._U_c<self._size:
            self._U_store[self._U_c] = val #float(not(math.isnan(val))) * val
            self._U_c+=1
            
    def K_append(self,val): 
        '''
        Append a value to kenetic energy.
        
        :arg double val: value to append
        '''
        
        if self._K_c<self._size:
            self._K_store[self._K_c] = val #float(not(math.isnan(val))) * val
            self._K_c+=1        
    def Q_append(self,val): 
        '''
        Append a value to total energy.
        
        :arg double val: value to append
        '''
        if self._Q_c<self._size:
            self._Q_store[self._Q_c] = val #float(not(math.isnan(val))) * val
            self._Q_c+=1
    def T_append(self,val):
        '''
        Append a value to time store.
        
        :arg double val: value to append
        '''
        if self._T_c<self._size: 
            self._T_store[self._T_c] = val + self._T_base
            self._T_c+=1            
   
    def plot(self):
        '''
        Plot recorded energies against time.
        '''
        
        if (self._Mh != None and self._Mh.nproc > 1):
            
            
            #data to collect
            _d = [self._Q_store.Dat, self._U_store.Dat, self._K_store.Dat]
            
            
            
            #make a temporary buffer.
            if self._Mh.rank == 0:
                _buff = ScalarArray(initial_value = 0.0, ncomp = self._T_store.ncomp, dtype=ctypes.c_double)
                _T = self._T_store.Dat
                _Q = ScalarArray(initial_value = 0.0, ncomp = self._T_store.ncomp, dtype=ctypes.c_double)
                _U = ScalarArray(initial_value = 0.0, ncomp = self._T_store.ncomp, dtype=ctypes.c_double)
                _K = ScalarArray(initial_value = 0.0, ncomp = self._T_store.ncomp, dtype=ctypes.c_double)
            
                _Q.Dat[::] += self._Q_store.Dat[::]
                _U.Dat[::] += self._U_store.Dat[::]
                _K.Dat[::] += self._K_store.Dat[::]
                
                _dl = [_Q.Dat, _U.Dat, _K.Dat]
            else:
                _dl = [None, None, None]
            
            
            
            for _di, _dj in zip(_d, _dl):
                
                if self._Mh.rank == 0:
                    _MS=MPI.Status()
                    for ix in range(1,self._Mh.nproc):
                        self._Mh.comm.Recv(_buff.Dat[::], ix, ix, _MS)
                        _dj[::] += _buff.Dat[::]
                        
                        
                else:
                    self._Mh.comm.Send(_di[::], 0, self._Mh.rank)
            
            if self._Mh.rank == 0:
                _Q = _Q.Dat
                _U = _U.Dat
                _K = _K.Dat
            
            
        
        else:
            _T = self._T_store.Dat
            _Q = self._Q_store.Dat
            _U = self._U_store.Dat
            _K = self._K_store.Dat
        
        
        
        
        if (self._Mh == None or self._Mh.rank == 0):
            print "last total" ,self._Q_store.Dat[self._Q_store.end]
            print "last kinetic", self._K_store.Dat[self._K_store.end]
            print "last potential", self._U_store.Dat[self._U_store.end]        
            print "============================================="
            print "first total" ,self._Q_store.Dat[0]
            print "first kinetic", self._K_store.Dat[0]
            print "first potential", self._U_store.Dat[0]
            
            #plt.ion()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            
            
            ax2.plot(_T,_Q,color='r', linewidth=2)
            ax2.plot(_T,_U,color='g')
            ax2.plot(_T,_K,color='b')
            
            ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kinetic energy')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Energy')
            
            fig2.canvas.draw()
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
    def __init__(self, initial_value = None, name = None, ncomp = 1, val = None, dtype=ctypes.c_double, max_size = None):
        '''
        Creates scalar with given initial value.
        '''
        
        if max_size == None:
            self._max_size = ncomp
        else:
            self._max_size = max_size
        
        
        
        
        self._dtype = dtype
        
        if (name != None):
            self._name = name
        self._N1 = ncomp
        
        if (initial_value != None):
            if (type(initial_value) is np.ndarray):
                self._Dat = np.array(initial_value, dtype=self._dtype, order='C')
                self._N1 = initial_value.shape[0]
                self._max_size = self._N1
            elif (type(initial_value) == list):
                self._Dat = np.array(np.array(initial_value), dtype=self._dtype, order='C')
                self._N1 = len(initial_value)
                self._max_size = self._N1
            else:
                self._Dat = float(initial_value) * np.ones([self._N1], dtype=self._dtype, order='C')
                
        elif (val == None):
            self._Dat = np.zeros([self._max_size], dtype=self._dtype, order='C')
        elif (val != None):
            self._Dat = np.array([val], dtype=self._dtype, order='C')
        
        self._A = False
        self._DatHaloInit = False
        
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
    
        
    def __getitem__(self,ix):
        return self._Dat[ix]
        
    def scale(self,val):
        '''
        Scale data array by value val.
        
        :arg double val: Coefficient to scale all elements by.
        '''
        #the below seems to cause glibc errors
        self._Dat = self._Dat * np.array([val],dtype=self._dtype)
        '''
        #work around
        if (self._DatHaloInit == False):
            _s = 1
        else:
            _s = 2
        
        
        for ix in range(_s*self._N1):
            self._Dat[ix]=val*self._Dat[ix]
        '''
        
        
        
    def zero(self):
        '''
        Zero all elements in array.
        '''
        #causes glibc errors
        self._Dat = np.zeros(self._N1, dtype=self._dtype, order='C')
        '''
        #work around
        for ix in range(self._N1):
            self._Dat[ix]=0.
        '''
    
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
        
    @ncomp.setter
    def ncomp(self,val):
        assert val <= self._max_size, "ncomp, max_size error"
        self._N1 = val
        
        
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
        return self._name
    
    def resize(self, N):
        if (N>self._max_size):
            self._max_size = N+(N-self._max_size)*10
            self._Dat = np.resize(self._Dat,self._max_size)
        #self._N1 = N
    
    @property
    def end(self):
        '''
        Returns end index of array.
        '''
        return self._max_size - 1
    
    @property
    def sum(self):
        '''
        Return array sum
        '''
        return self._Dat.sum()
    
        
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
            
    def InitHaloDat(self):
        '''
        Create a secondary dat container.
        '''
        
        if(self._DatHaloInit == False):
            self._max_size = 2*self._max_size
            self._Dat = np.resize(self._Dat,self._max_size)
            self._DatHaloInit = True
            
    
    @property
    def DatHaloInit(self):
        '''
        Return status of halo dat.
        '''
        return self._DatHaloInit
        
        
    
    
                       
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

    @property
    def ncomp(self):
        '''Return number of components'''
        return self._length

    def __getitem__(self,ix):
        return self._Dat[ix]

    def __setitem__(self,ix,val):
        self._Dat[ix] = val

################################################################################################################
# Blank arrays.
################################################################################################################ 

NullIntScalarArray=ScalarArray(dtype=ctypes.c_int)
NullDoubleScalarArray=ScalarArray(dtype=ctypes.c_double)














        
        
