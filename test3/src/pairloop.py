import numpy as np
import particle
import math
import ctypes
import time
import random
import os
import hashlib
import subprocess

class PairLoopRapaport():
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
        
        
        
    def _arg_update(self):    
        self._args = [self._cell_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self._q_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self._input_state.positions().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.domain().extent().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.accelerations().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.U().ctypes.data_as(ctypes.POINTER(ctypes.c_double))]
        
        
    
        
        
     
    def _update_prepare(self):
        #handle perodic bounadies
        self._input_state.domain().boundary_correct(self._input_state)
        #update cell list
        self.cell_sort_all()
        
        
    def update(self):
        """
        C version of the pair_locate: Loop over all cells update accelerations and potential engery.
        """
        self._update_prepare()
    
        self._input_state.set_accelerations(0.0)
        self._input_state.reset_U() #causes segfault.....
        
        self._arg_update()
            
        self._libpair_loop_LJ.d_pair_loop_LJ(ctypes.c_int(self._input_state.N()), ctypes.c_int(self._input_state.domain().cell_count()), ctypes.c_double(self._input_state._potential._rc), *self._args)   
        
        
        
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
        

class SingleAllParticleLoop():
    """
    Class to loop over all particles once.
    """
    def __init__(self, kernel, particle_dat_dict, headers=None):
        self._temp_dir = './build/'
        if (not os.path.exists(self._temp_dir)):
            os.mkdir(self._temp_dir)
        self._kernel = kernel
        self._particle_dat_dict = particle_dat_dict
        self._nargs = len(self._particle_dat_dict)
        self._headers = headers
        
        self._unique_name = self._unique_name_calc()
        
        self._lib_filename  = self._unique_name +'.so'
        
        if (not os.path.exists(os.path.join(self._temp_dir,self._library_filename))):
            self._create_library()



    def _create_library(self):
        '''
        Create a shared library from the source code.
        '''
        
        filename_base = os.path.join(self._temp_dir,self._unique_name)
        header_filename = filename_base+'.h'
        impl_filename = filename_base+'.c'
        with open(header_filename,'w') as f:
            print >> f, self._generate_header_source()        
        
        
        
        
        
        
        
        
        
    def _generate_header_source(self):
        '''Generate the source code of the header file.

        Returns the source code for the header file.
        '''
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME':self._unique_name,
             'INCLUDED_HEADERS':self._included_headers(),
             'KERNEL_NAME':self._kernel.name,
             'ARGUMENTS':self._argnames()}
        return (code % d)



    def _unique_name_calc(self):
        '''Return name which can be used to identify the pair loop 
        in a unique way.
        '''
        return self._kernel.name+'_'+self.hexdigest()
        
    def hexdigest(self):
        '''Create unique hex digest'''
        m = hashlib.md5()
        m.update(self._kernel.code)
        if (self._headers != None):
            for header in self._headers:
                m.update(header)
        return m.hexdigest()

























        
    
    
