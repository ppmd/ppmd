import numpy as np
import particle
import math
import ctypes
import time
import random

class pair_loop_rapaport():
    '''
    Class to implement rapaport 14 cell looping.
    '''
    def __init__(self,input_state):
        
        
        
        self._input_state = input_state
        
        '''Construct initial cell list'''
        self._q_list = np.zeros([1 + self._input_state.N() + self._input_state.domain().cell_count()], dtype=ctypes.c_int, order='C')
        self.cell_sort_all()
        

        
        '''Determine cell neighbours'''
        self._cell_map=np.zeros([14*self._input_state.domain().cell_count(),5],dtype=ctypes.c_int, order='C')
        for cp in range(1,1 + self._input_state.domain().cell_count()):
            self._cell_map[(cp-1)*14:(cp*14),...] = self.get_adjacent_cells(cp)
        
        
        
        '''Initialise pair_loop code'''
        self._libpair_loop_LJ = np.ctypeslib.load_library('libpair_loop_LJ.so','.')
        self._libpair_loop_LJ.d_pair_loop_LJ.restype = ctypes.c_int
        
        #void d_pair_loop_LJ(int N, int cell_count, double rc, int* cells, int* q_list, double* pos, double* d_extent, double *accel);
        self._libpair_loop_LJ.d_pair_loop_LJ.argtypes = [ctypes.c_int,
                                                        ctypes.c_int,
                                                        ctypes.c_double,
                                                        ctypes.POINTER(ctypes.c_int),
                                                        ctypes.POINTER(ctypes.c_int),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double)]
        
           
        self._args = [self._cell_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self._q_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self._input_state.positions().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.domain().extent().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.accelerations().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.U().ctypes.data_as(ctypes.POINTER(ctypes.c_double))]
        
        
    
        
        
     
    def update(self):
        #handle perodic bounadies
        self._input_state.domain().boundary_correct(self._input_state)
        #update cell list
        self.cell_sort_all()
        
        
    def pair_locate_c(self):
        """
        C version of the pair_locate: Loop over all cells update accelerations and potential engery.
        """
    
        self._input_state.set_accelerations(0.0)
        #self._input_state.reset_U() #causes segfault.....
        
            
        self._libpair_loop_LJ.d_pair_loop_LJ(ctypes.c_int(self._input_state.N()), ctypes.c_int(self._input_state.domain().cell_count()), ctypes.c_double(self._input_state._rc), *self._args)   
        
        
        
    def cell_sort_all(self):
        """
        Construct neighbour list, assigning atoms to cells. Using Rapaport alg.
        """
        for cx in range(1,1+self._input_state.domain().cell_count()):
            self._q_list[self._input_state.N() + cx] = 0
        for ix in range(1,1+self._input_state.N()):
            c = self._input_state.domain().get_cell_lin_index(self._input_state.positions()[ix-1,])
            
            #print c, self._pos[ix-1,], self._domain._extent*0.5
            self._q_list[ix] = self._q_list[self._input_state.N() + c]
            self._q_list[self._input_state.N() + c] = ix
            
      
        
      
       
    def get_adjacent_cells(self,ix):
        """
        Returns the 14 neighbouring cells as linear index.
        
        :arg ix: (int) Input index.
        
        """
         
        cell_list = np.zeros([14,5],dtype=ctypes.c_int, order='C')
        #cell_list_boundary=[]
        
        C = self._input_state.domain().cell_index_tuple(ix)
        
        stencil_map = [
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [1,-1,0],
            [-1,1,1],
            [0,1,1],
            [1,1,1],
            [-1,0,1],
            [0,0,1],
            [1,0,1],
            [-1,-1,1],
            [0,-1,1],
            [1,-1,1]
            ]
        
        for ix in range(14):
            ind = stencil_map[ix]
            
            cell_list[ix,] = self._input_state.domain().cell_index_lin_offset(C+ind)

        return cell_list        
        
        
    
    
