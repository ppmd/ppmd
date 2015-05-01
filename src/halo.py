import numpy as np
import math
import ctypes
import data
from mpi4py import MPI
import random

################################################################################################################
# HALO DEFINITIONS
################################################################################################################
             
class HaloCartesian(object):
    """
    Class to contain and control cartesian halo transfers.
    
    """
    def __init__(self, MPICOMM = None, rank = 0, nproc = 1, cell_array = None):
        assert cell_array != None, "Error: No cell array passed."
        self._rank = rank
        self._nproc = nproc
        self._MPI = MPICOMM
        
        self._ca = cell_array
        self._halos=[]
        
        _BASE_SIZES = [1, self._ca[0], 1, self._ca[1], self._ca[0]*self._ca[1], self._ca[1],1,self._ca[0],1]
        
        _MID_SIZES = [self._ca[2], self._ca[0]*self._ca[2], self._ca[2], self._ca[1]*self._ca[2], self._ca[1]*self._ca[2], self._ca[2], self._ca[0]*self._ca[2], self._ca[2]]
        
        self._SIZES = _BASE_SIZES + _MID_SIZES + _BASE_SIZES
        
        dest=0
        src=0
        
        for ix in range(26):
            self._halos.append(Halo(self._MPI, self._rank,dest,src,ix,self._SIZES[ix]))
           
            
    def exchange(self,cell_contents_count, cell_list, data):
        '''Get new storage sizes'''
        self._exchange_size_calc(cell_contents_count)
        '''Resize'''
        for i,h in enumerate(self._halos):
            h.send_prepare(self._exchange_sizes[i], cell_list, data)
    
    
        
    def _local_data_pack(self):
        pass
        
        
        
        
        
    def _exchange_size_calc(self,cell_contents_count):
        
        end=self._ca[0]*self._ca[1]*self._ca[2]
        
        #TOP
        self._cell_indices=[
                            [end-1],
                            range(self._ca[0]*(self._ca[1]*self._ca[2]-1),end,1),
                            [self._ca[0]*self._ca[1]*self._ca[2]-self._ca[0]],
                            range(self._ca[0]*(self._ca[1]*(self._ca[2]-1)+1)-1,end,self._ca[0]),
                            range(self._ca[0]*self._ca[1]*(self._ca[2]-1),end,1),
                            range(self._ca[0]*self._ca[1]*(self._ca[2]-1),end,self._ca[0]),
                            [self._ca[0]*(self._ca[1]*(self._ca[2]-1)+1)-1],
                            range(self._ca[0]*self._ca[1]*(self._ca[2]-1),self._ca[0]*self._ca[1]*(self._ca[2]-1)+self._ca[0],1),
                            [self._ca[0]*self._ca[1]*(self._ca[2]-1)]
                            ]
                            
                            
                            
        
        #bottom layer
        sizes=[
                cell_contents_count[self._ca[0]*self._ca[1]*self._ca[2]-1],
                sum(cell_contents_count[self._ca[0]*(self._ca[1]*self._ca[2]-1)::1]),
                cell_contents_count[self._ca[0]*self._ca[1]*self._ca[2]-self._ca[0]], 
                sum(cell_contents_count[self._ca[0]*(self._ca[1]*(self._ca[2]-1)+1)-1::self._ca[0]]),
                sum(cell_contents_count[self._ca[0]*self._ca[1]*(self._ca[2]-1)::1]),
                sum(cell_contents_count[self._ca[0]*self._ca[1]*(self._ca[2]-1)::self._ca[0]]),#
                cell_contents_count[self._ca[0]*(self._ca[1]*(self._ca[2]-1)+1)-1],
                sum(cell_contents_count[self._ca[0]*self._ca[1]*(self._ca[2]-1):self._ca[0]*self._ca[1]*(self._ca[2]-1)+self._ca[0]:1]),
                cell_contents_count[self._ca[0]*self._ca[1]*(self._ca[2]-1)]
              ]
        
        
      
        
        
        #middle layer
        sizes+=[
               sum(cell_contents_count[self._ca[0]*self._ca[1]-1::self._ca[0]*self._ca[1] ])]
        
        
        tmp1=0
        tmp1_=[]
        for ix in range(1,self._ca[2]+1):
            tmp1_+=range(ix*self._ca[0]*self._ca[1]-self._ca[0],ix*self._ca[0]*self._ca[1],1)
            tmp1+=sum(cell_contents_count[ix*self._ca[0]*self._ca[1]-self._ca[0]:ix*self._ca[0]*self._ca[1]:1])       
            
        
        
        tmp2=0
        tmp2_=[]
        for ix in range(self._ca[2]):
            tmp2_+= range(ix*self._ca[0]*self._ca[1],ix*self._ca[0]*self._ca[1]+self._ca[0],1)
            tmp2+=sum(cell_contents_count[ix*self._ca[0]*self._ca[1]:ix*self._ca[0]*self._ca[1]+self._ca[0]-1:1])
        
        
        self._cell_indices+=[
                              range(self._ca[0]*self._ca[1]-1,end,self._ca[0]*self._ca[1]),
                              tmp1_,
                              range(self._ca[0]*(self._ca[1]-1),self._ca[0]*(self._ca[1]*self._ca[2]-1)+1,self._ca[0]*self._ca[1]),
                              range(self._ca[0]-1,end,self._ca[0]),
                              range(0,end,self._ca[0]),
                              range(self._ca[0]-1,end,self._ca[0]*self._ca[1]),
                              tmp2_,
                              range(0,end,self._ca[0]*self._ca[1])
                             ]
        
        
        sizes+=[
                tmp1,
                sum(cell_contents_count[self._ca[0]*(self._ca[1]-1):self._ca[0]*(self._ca[1]*self._ca[2]-1)+1:self._ca[0]*self._ca[1]]),
                sum(cell_contents_count[self._ca[0]-1::self._ca[0]]),
                sum(cell_contents_count[0::self._ca[0]]),#
                sum(cell_contents_count[self._ca[0]-1::self._ca[0]*self._ca[1]]),
                tmp2,
                sum(cell_contents_count[0::self._ca[0]*self._ca[1]])
               ]
        
        
        
        self._cell_indices+=[
                            [self._ca[0]*self._ca[1]-1],
                            range(self._ca[0]*(self._ca[1]-1),self._ca[0]*self._ca[1],1),
                            [self._ca[0]*(self._ca[1]-1)],
                            range(self._ca[0]-1,self._ca[0]*self._ca[1],self._ca[0]),
                            range(0,self._ca[0]*self._ca[1],1),
                            range(0,self._ca[0]*(self._ca[1]-1)+1,self._ca[0]),
                            [self._ca[0]-1],
                            range(0,self._ca[0],1),
                            [0]
                            ]
        
        
        
        #Top layer
        sizes+=[
                cell_contents_count[self._ca[0]*self._ca[1]-1],
                sum(cell_contents_count[self._ca[0]*(self._ca[1]-1):self._ca[0]*self._ca[1]:1]),
                cell_contents_count[self._ca[0]*(self._ca[1]-1)],
                sum(cell_contents_count[self._ca[0]-1:self._ca[0]*self._ca[1]:self._ca[0]]),
                sum(cell_contents_count[0:self._ca[0]*self._ca[1]:1]),
                sum(cell_contents_count[0:self._ca[0]*(self._ca[1]-1)+1:self._ca[0]]),
                cell_contents_count[self._ca[0]-1],
                sum(cell_contents_count[0:self._ca[0]:1]),
                cell_contents_count[0]
              ]
        
        
        
        #print "########################################################"
        #print self._cell_indices
        
        _sizes=[]
        
        for i,x in enumerate(self._cell_indices):
            _sizes.append(sum([y[1] for y in enumerate(cell_contents_count) if y[0] in x]))
        
        
        
        
        
        
        
        
                
        self._exchange_sizes = sizes
        
        
        
        
    
    
class Halo(object):
    """
    Class to contain a halo.
    """
    def __init__(self, MPICOMM = None, rank_local = 0, rank_dest = 0, rank_src = 0, local_index = None, cell_count = 1, nrow = 1, ncol = 1, dtype = ctypes.c_double):
        
        assert local_index != None, "Error: No local index specified."
        
        self._MPI = MPICOMM
        self._rank = rank_local
        self._rd = rank_dest
        self._rs = rank_src
        self._MPIstatus=MPI.Status()
        
        
        self._li = local_index #make linear or 3 tuple? leading towards linear.
        self._nc = ncol
        self._nr = nrow
        self._dt = dtype
        self._cell_count = cell_count
        
        self._d = np.empty((self._nr, self._nc), dtype=self._dt, order='C')
    
    def resize(self, nrow = None, ncol = None):
        """
        Resize halo to given dimensions. If a new dimension size is not given, old dimension size will remain. Initial testing gives small hit. 
        
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
            self._d = np.empty((self._nr, self._nc), dtype=self._dt, order='C')
    
    
    
    
    def send_prepare(self, count, cell_list, data):
        
        '''Exchange sizes of new data.'''
        recv_size = np.array([0],dtype='i')
        self._MPI.Sendrecv(np.array([count],dtype='i'), self._rd, self._rd, recv_size, self._rs, self._rs, self._MPIstatus)
        
        '''Create space for new incoming data.'''
        self.resize(recv_size[0],data.ncomp)
        
        '''Create space for packing outgoing data.'''
        self._send_buffer = np.empty((count, self._nc), dtype=self._dt, order='C')
        
        
        
        
        
    
        
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
    
    
    
    
    
    
    
    
    
    
    
