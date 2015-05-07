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
        
        _BASE_SIZES = [1, self._ca[0]-2, 1, self._ca[1]-2, (self._ca[0]-2)*(self._ca[1]-2), self._ca[1]-2,1,self._ca[0]-2,1]
        
        _MID_SIZES = [self._ca[2]-2, (self._ca[0]-2)*(self._ca[2]-2), self._ca[2]-2, (self._ca[1]-2)*(self._ca[2]-2), (self._ca[1]-2)*(self._ca[2]-2), self._ca[2]-2, (self._ca[0]-2)*(self._ca[2]-2), self._ca[2]-2]
        
        self._SIZES = _BASE_SIZES + _MID_SIZES + _BASE_SIZES
        
        dest=0
        src=0
        
        self._exchange_prepare()
        
        for ix in range(26):
            self._halos.append(Halo(self._MPI, self._rank,dest,src,ix,self._SIZES[ix],self._cell_indices[ix]))
            
        
           
        
        
    def exchange(self,cell_contents_count, cell_list, data):
        '''Get new storage sizes'''
        self._exchange_size_calc(cell_contents_count)
        
        '''Reset halo starting points'''
        data.halo_start_reset()
        
        
        for i,h in enumerate(self._halos):
            h.send_prepare(self._exchange_sizes[i], cell_list, data)
            
    
        
    def _local_data_pack(self):
        pass
        
        
        
        
        
    def _exchange_size_calc(self,cell_contents_count):
        
        for i,x in enumerate(self._cell_indices):
            self._exchange_sizes[i]=(sum([y[1] for y in enumerate(cell_contents_count) if y[0] in x]))
        
        
    def _exchange_prepare(self):
        
        
        _E = self._ca[0]*self._ca[1]*(self._ca[2]-1) - self._ca[0] - 1
        _TS = _E - self._ca[0]*(self._ca[1] - 2) + 2
        
        _BE = self._ca[0]*(2*self._ca[1] - 1) - 1
        _BS = self._ca[0]*(self._ca[1] + 1) + 1
        
        _tmp4=[]
        for ix in range(self._ca[1]-2):
            _tmp4 += range(_TS + ix*self._ca[0], _TS + (ix+1)*self._ca[0] - 2, 1)
        
        

        self._cell_indices=[
                            [_E-1],
                            range(_E-self._ca[0]+2,_E,1),
                            [_E-self._ca[0]+2],
                            range(_TS+self._ca[0]-3, _E, self._ca[0]), 
                            _tmp4, 
                            range(_TS,_E,self._ca[0]), 
                            [_TS+self._ca[0]-3], 
                            range(_TS,_TS+self._ca[0]-2,1),
                            [_TS]
                            ]
        
        #MIDDLE
        _tmp10=[]
        for ix in range(self._ca[2]-2):
            _tmp10 += range( _BE-self._ca[0]+2 + ix*self._ca[0]*self._ca[1], _BE + ix*self._ca[0]*self._ca[1],1)
        
        _tmp12=[]
        for ix in range(self._ca[2]-2):
            _tmp12 += range( _BS+self._ca[0]-3 + ix*self._ca[0]*self._ca[1], _BE + ix*self._ca[0]*self._ca[1], self._ca[0])
        
        _tmp13=[]
        for ix in range(self._ca[2]-2):
            _tmp13 += range( _BS + ix*self._ca[0]*self._ca[1], _BE + ix*self._ca[0]*self._ca[1], self._ca[0])        
        
        _tmp15=[]
        for ix in range(self._ca[2]-2):
            _tmp15 += range( _BS + ix*self._ca[0]*self._ca[1], _BS+self._ca[0]-2 + ix*self._ca[0]*self._ca[1], 1)

        
        
        self._cell_indices+=[
                              range(_BE - 1,_E,self._ca[0]*self._ca[1]),
                              _tmp10,
                              range(_BE-self._ca[0]+2,_E,self._ca[0]*self._ca[1]),
                              _tmp12,
                              _tmp13,
                              range(_BS+self._ca[0]-3,_E,self._ca[0]*self._ca[1]),
                              _tmp15,
                              range(_BS,_E,self._ca[0]*self._ca[1])
                             ]
        
        _tmp21=[]
        for ix in range(self._ca[1]-2):
            _tmp21 += range(_BS + ix*self._ca[0], _BS + (ix+1)*self._ca[0] - 2, 1)
                    
        #BOTTOM
        self._cell_indices+=[
                            [_BE-1],
                            range(_BE-self._ca[0]+2,_BE,1),
                            [_BE-self._ca[0]+2],
                            range(_BS+self._ca[0]-3, _BE, self._ca[0]), 
                            _tmp21,
                            range(_BS,_BE,self._ca[0]), 
                            [_BS+self._ca[0]-3], 
                            range(_BS,_BS+self._ca[0]-2,1),
                            [_BS]
                            ]
        
        self._exchange_sizes=range(26)             
        
        
        
        
    
    
class Halo(object):
    """
    Class to contain a halo.
    """
    def __init__(self, MPICOMM = None, rank_local = 0, rank_dest = 0, rank_src = 0, local_index = None, cell_count = 1, cell_indices = None, nrow = 1, ncol = 1, dtype = ctypes.c_double):
        
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
        
        if (cell_indices!=None):
            self._cell_indices = data.ScalarArray(cell_indices, dtype=ctypes.c_int)
        
        
        
        _code = '''
        int index = 0;
        
        for(int ic=0;ic<num_cells;ic++){
                            
            const int icp = cell_indices[ic];               
            int ix = cell_list[npart+icp];
            while (ix > -1){
                
                for(int iy=0;iy<ncomp;iy++){
                    send_buffer[LINIDX_2D(ncomp,index,iy)] = data_buffer[LINIDX_2D(ncomp,ix,iy)];
                }
                
                index++;
                ix=cell_list[ix];
                
                }} '''
         #
        _static_args = {'num_cells':ctypes.c_int, 
                        'npart':ctypes.c_int, 
                        'ncomp':ctypes.c_int}
        
        _args = {'cell_indices':data.NullIntScalarArray, 
                 'cell_list':data.NullIntScalarArray,
                 'send_buffer':data.NullDoubleScalarArray, 
                 'data_buffer':data.NullDoubleScalarArray}
                 
        _headers = ['stdio.h']
        _kernel = kernel.Kernel('HaloPack', _code, None, _headers, None, _static_args)
        self._packing_lib = build.SharedLib(_kernel,_args,True)
        
        self._send_buffer = particle.Dat(1000, self._nc, name='send_buffer', dtype=ctypes.c_double)
    
    
    
    
    def set_cell_indices(self, cell_indices):
        self._cell_indices = data.ScalarArray(cell_indices, dtype=ctypes.c_int)
    
    
    
    def send_prepare(self, count, cell_list, data_buffer, cell_indices = None):
          
        
        '''Loop over the local cells and collect particle data using the cell list and list of cell indices'''
        
        if (cell_indices!=None):
            self._cell_indices = data.ScalarArray(cell_indices, dtype=ctypes.c_int)        
        
        self._packing_lib.execute( {'cell_indices':self._cell_indices, 
                                    'cell_list':cell_list,
                                    'send_buffer':self._send_buffer, 
                                    'data_buffer':data_buffer} , 
                     static_args = {'num_cells':ctypes.c_int(self._cell_count),
                                    'npart':ctypes.c_int(data_buffer.npart),
                                    'ncomp':ctypes.c_int(data_buffer.ncomp) })
    
        
        '''Send data'''
        
        
        
        self._MPI.Sendrecv(self._send_buffer.Dat[0:count+1:1][::], self._rd, self._rd, data_buffer.Dat[data_buffer.halo_start,::], self._rs, self._rs, self._MPIstatus)
        
        _shift=self._MPIstatus.Get_count( data.mpi_map[data_buffer.dtype])-1
        
        data_buffer.halo_start_shift(_shift)
        
        
        
        
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
    
    
    
    
    
    
    
    
    
    
    
