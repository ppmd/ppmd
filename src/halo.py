import numpy as np
import math
import ctypes
import data
from mpi4py import MPI
import random
import kernel
import build
import particle

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
        
        #currently sends wrong cell indices, needs to send the cell indices to pack.
        
        for i,h in enumerate(self._halos):
            h.send_prepare(self._exchange_sizes[i], self._cell_indices[i], cell_list, data)
    
    
        
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
        
        #MIDDLE
        _tmp1=[]
        for ix in range(1,self._ca[2]+1):
            _tmp1+=range(ix*self._ca[0]*self._ca[1]-self._ca[0],ix*self._ca[0]*self._ca[1],1)    
        
        _tmp2=[]
        for ix in range(self._ca[2]):
            _tmp2+= range(ix*self._ca[0]*self._ca[1],ix*self._ca[0]*self._ca[1]+self._ca[0],1)
        
        
        self._cell_indices+=[
                              range(self._ca[0]*self._ca[1]-1,end,self._ca[0]*self._ca[1]),
                              _tmp1,
                              range(self._ca[0]*(self._ca[1]-1),self._ca[0]*(self._ca[1]*self._ca[2]-1)+1,self._ca[0]*self._ca[1]),
                              range(self._ca[0]-1,end,self._ca[0]),
                              range(0,end,self._ca[0]),
                              range(self._ca[0]-1,end,self._ca[0]*self._ca[1]),
                              _tmp2,
                              range(0,end,self._ca[0]*self._ca[1])
                             ]
        
        
        #BOTTOM
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
        
        _sizes=range(self._ca[0]*self._ca[1]*self._ca[2])
        
        for i,x in enumerate(self._cell_indices):
            _sizes[i]=(sum([y[1] for y in enumerate(cell_contents_count) if y[0] in x]))
        
                
        self._exchange_sizes = _sizes
        
        
        
        
    
    
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
        
        self._packing_lib = None
        
    
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
            self._d = particle.Dat(self._nr, self._nc, dtype=self._dt)
    
    
    
    
    def send_prepare(self, count, cell_indices, cell_list, data_buffer):
        
        '''Exchange sizes of new data.'''
        recv_size = np.array([0],dtype='i')
        self._MPI.Sendrecv(np.array([count],dtype='i'), self._rd, self._rd, recv_size, self._rs, self._rs, self._MPIstatus)
        
        '''Create space for new incoming data.'''
        self.resize(recv_size[0],data_buffer.ncomp)
        
        '''Create space for packing outgoing data.'''
        #self._send_buffer = np.empty((count, self._nc), dtype=self._dt, order='C')
        self._send_buffer = particle.Dat(count, self._nc, name='send_buffer', dtype=ctypes.c_double)
        
        '''Loop over the local cells and collect particle data using the cell list and list of cell indices'''
        
        
        '''MOVE THIS TO C, TAKES YEARS, maybe one loop for all send buffers via pointer to pointers as opposed to here which is a loop per halo
        index=0
        for ic in cell_indices:
            ix = cell_list[data_buffer.npart+ic]
            while (ix > -1):
                
                self._send_buffer[index][:]=data_buffer[ix]
                
                index+=1
                ix=cell_list[ix];
        '''
        

        
        _cell_indices = data.ScalarArray(initial_value=cell_indices, dtype=ctypes.c_int)
        
        
        if (self._packing_lib == None):
            _d = {'LOOP_UNROLL':build.loop_unroll('send_buffer[LINIDX_2D(ncomp,index,iy)] = data_buffer[LINIDX_2D(ncomp,ix,iy)];',0,data_buffer.ncomp-1,1,'','iy')}
            _code = '''
            int index = 0;
            
            for(int ic=0;ic<num_cells;ic++){
                                
                const int icp = cell_indices[ic];               
                int ix = cell_list[npart+icp];
                while (ix > -1){
                    
                    %(LOOP_UNROLL)s
                    
                    index++;
                    ix=cell_list[ix];
                    
                    }} ''' % _d
             #
            _static_args = {'num_cells':ctypes.c_int, 
                            'npart':ctypes.c_int, 
                            'ncomp':ctypes.c_int}
            
            _args = {'cell_indices':_cell_indices, 
                     'cell_list':cell_list,
                     'send_buffer':self._send_buffer, 
                     'data_buffer':data_buffer}
                     
            _headers = ['stdio.h']
            _kernel = kernel.Kernel('HaloPack', _code, None, _headers, None, _static_args)
            self._packing_lib = build.SharedLib(_kernel,_args,True)
            
        
        
        self._packing_lib.execute( {'cell_indices':_cell_indices, 
                                    'cell_list':cell_list,
                                    'send_buffer':self._send_buffer, 
                                    'data_buffer':data_buffer} , 
                     static_args = {'num_cells':ctypes.c_int(self._cell_count),
                                    'npart':ctypes.c_int(data_buffer.npart),
                                    'ncomp':ctypes.c_int(data_buffer.ncomp) })
    
    
        
    
        
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
        '''Return number of rows.'''    
        return self._nr
        
    @property
    def index(self):
        '''Return local index.'''
        return self._li
        
    @property
    def dtype(self):
        ''' Return Dat c data ctype'''
        return self._dt
    
    
    
    
    
    
    
    
    
    
    
