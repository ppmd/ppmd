import numpy as np
import particle
import math
import ctypes
import random
import os
import hashlib
import subprocess
import data
import kernel
import loop
import constant
import build



class _base(build.GenericToolChain):
    
    
    def _kernel_methodname(self):
        '''Construct the name of the kernel method.
        
        Return a string of the form 
        ``inline void kernel_name(double **<arg1>, double *<arg2}, ...) {``
        which is used for defining the name of the kernel method.
        '''
        space = ' '*14
        s = 'inline void '+self._kernel.name+'('
        
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            if (type(dat[1]) == particle.Dat):
                s += data.ctypes_map[dat[1].dtype]+' **'+dat[0]+', '
            if (type(dat[1]) == data.ScalarArray):
                s += data.ctypes_map[dat[1].dtype]+' *'+dat[0]+', '
            
        
        s = s[:-2] + ') {'
        return s           

    
    def _kernel_argument_declarations(self):
        '''Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a 
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as 
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        '''
        s = '\n'
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            
            space = ' '*14
            argname = dat[0]+'_ext'
            loc_argname = dat[0]
            
            
            if (type(dat[1]) == data.ScalarArray):
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+' = '+argname+';\n'
            
            if (type(dat[1]) == particle.Dat):
                ncomp = dat[1].ncomp
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+'[2];\n'
                s += space+loc_argname+'[0] = '+argname+'+'+str(ncomp)+'*i;\n'
                s += space+loc_argname+'[1] = '+argname+'+'+str(ncomp)+'*j;\n'       
        
        return s 

################################################################################################################
# RAPAPORT LOOP SERIAL
################################################################################################################

class PairLoopRapaport(_base):
    '''
    Class to implement rapaport 14 cell looping.
    
    :arg int N: Number of elements to loop over.
    :arg domain domain: Domain containing the particles.
    :arg dat positions: Postitions of particles.
    :arg potential potential: Potential between particles.
    :arg dict dat_dict: Dictonary mapping between state vars and kernel vars.
    :arg bool DEBUG: Flag to enable debug flags.
    '''
    def __init__(self,N,domain,positions,potential,dat_dict, DEBUG = False):
        self._DEBUG = DEBUG
        self._N = N
        self._domain = domain
        self._P = positions
        self._potential = potential
        self._particle_dat_dict = dat_dict
        
        self._compiler_set()
        
        
        ##########
        # End of Rapaport initialisations.
        ##########
        
        self._temp_dir = './build/'
        if (not os.path.exists(self._temp_dir)):
            os.mkdir(self._temp_dir)
        self._kernel = self._potential.kernel()
        
        
        self._nargs = len(self._particle_dat_dict)
        

        self._code_init()
        self._cell_sort_setup()
        
        self._unique_name = self._unique_name_calc()
        
        self._library_filename  = self._unique_name +'.so'
        
        if (not os.path.exists(os.path.join(self._temp_dir,self._library_filename))):
            self._create_library()
        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name,type(self))
    
    def _compiler_set(self):
        self._cc = build.TMPCC
    
                
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        
        
        inline void cell_index_offset(const unsigned int cp, const unsigned int cpp_i, int* cell_array, double *d_extent, unsigned int* cpp, unsigned int *flag, double *offset){
        
            const int cell_map[14][3] = {   {0,0,0},
                                            {1,0,0},
                                            {0,1,0},
                                            {1,1,0},
                                            {1,-1,0},
                                            {-1,1,1},
                                            {0,1,1},
                                            {1,1,1},
                                            {-1,0,1},
                                            {0,0,1},
                                            {1,0,1},
                                            {-1,-1,1},
                                            {0,-1,1},
                                            {1,-1,1}};     
                
                
            unsigned int tmp = cell_array[0]*cell_array[1];    
            int Cz = cp/tmp;
            int Cx = cp %% cell_array[0];
            int Cy = (cp - Cz*tmp)/cell_array[0];
            
            
            Cx += cell_map[cpp_i][0];
            Cy += cell_map[cpp_i][1];
            Cz += cell_map[cpp_i][2];
            
            int C0 = (Cx + cell_array[0]) %% cell_array[0];    
            int C1 = (Cy + cell_array[1]) %% cell_array[1];
            int C2 = (Cz + cell_array[2]) %% cell_array[2];
                
             
            if ((Cx != C0) || (Cy != C1) || (Cz != C2)) { 
                *flag = 1;
                offset[0] = ((double)sign(Cx - C0))*d_extent[0];
                offset[1] = ((double)sign(Cy - C1))*d_extent[1];
                offset[2] = ((double)sign(Cz - C2))*d_extent[2];
                
                
                
            } else {*flag = 0; }
            
            *cpp = (C2*cell_array[1] + C1)*cell_array[0] + C0;
            
            
                
            return;      
        }    
        
        void %(KERNEL_NAME)s_wrapper(const int n, const int cell_count, int* cell_array, int* q_list, double* d_extent,%(ARGUMENTS)s) { 
            
            
            for(unsigned int cp = 0; cp < cell_count; cp++){
                for(unsigned int cpp_i=0; cpp_i<14; cpp_i++){
                
                    double s[3]; 
                    unsigned int flag, cpp; 
                    int i,j;
                    
                    cell_index_offset(cp, cpp_i, cell_array, d_extent, &cpp, &flag, s);
                    
                    
                    double r1[3];
                    
                    i = q_list[n+cp];
                    while (i > -1){
                        j = q_list[n+cpp];
                        while (j > -1){
                            if (cp != cpp || i < j){
        
                                %(KERNEL_ARGUMENT_DECL)s
                                
                                  //KERNEL CODE START
                                  
                                  %(KERNEL)s
                                  
                                  //KERNEL CODE END
                                
                                
                            }
                            j = q_list[j];  
                        }
                        i=q_list[i];
                    }
                }
            }
            
            
            return;
        }        
        
        
        '''
    def _kernel_argument_declarations(self):
        '''Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a 
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as 
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        '''
        s = '\n'
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            
            space = ' '*14
            argname = dat[0]+'_ext'
            loc_argname = dat[0]
            
            
            if (type(dat[1]) == data.ScalarArray):
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+' = '+argname+';\n'
            
            if (type(dat[1]) == particle.Dat):
                if (dat[1].name  == 'positions'):
                    s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+'[2];\n'
                    
                    
                    s += space+'if (flag){ \n'

                    #s += space+'double r1[3];\n'
                    s += space+'r1[0] ='+argname+'[LINIDX_2D(3,j,0)] + s[0]; \n'
                    s += space+'r1[1] ='+argname+'[LINIDX_2D(3,j,1)] + s[1]; \n'
                    s += space+'r1[2] ='+argname+'[LINIDX_2D(3,j,2)] + s[2]; \n'
                    s += space+loc_argname+'[1] = r1;\n'
                    
                    s += space+'}else{ \n'
                    s += space+loc_argname+'[1] = '+argname+'+3*j;\n' 
                    s += space+'} \n'
                    s += space+loc_argname+'[0] = '+argname+'+3*i;\n'
                    
                else:
                    ncomp = dat[1].ncomp
                    s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+'[2];\n'
                    s += space+loc_argname+'[0] = '+argname+'+'+str(ncomp)+'*i;\n'
                    s += space+loc_argname+'[1] = '+argname+'+'+str(ncomp)+'*j;\n'       
        
        return s       
        
    def _generate_header_source(self):
        '''Generate the source code of the header file.

        Returns the source code for the header file.
        '''
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include "../generic.h"
        
        void %(KERNEL_NAME)s_wrapper(const int n,const int cell_count, int* cells, int* q_list, double* d_extent,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME':self._unique_name,
             'INCLUDED_HEADERS':self._included_headers(),
             'KERNEL_NAME':self._kernel.name,
             'ARGUMENTS':self._argnames()}
        return (code % d)        
                
        
    def execute(self, dat_dict = None):
        '''
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        '''
        self._cell_sort_all()
        
        if (dat_dict != None):
            self._particle_dat_dict = dat_dict
        
        args = [self._domain.cell_array.ctypes_data,
                        self._q_list.ctypes_data,
                        self._domain.extent.ctypes_data]
                
        for dat in self._particle_dat_dict.values():
            args.append(dat.ctypes_data)
            
        method = self._lib[self._kernel.name+'_wrapper']
        
        method(ctypes.c_int(self._N), ctypes.c_int(self._domain.cell_count), *args)           
    
    
    
    def _cell_sort_setup(self):
        """
        Creates looping for cell list creation
        """
        
        '''Construct initial cell list'''
        self._q_list = data.ScalarArray(np.zeros([self._N + self._domain.cell_count], dtype=ctypes.c_int, order='C'), dtype=ctypes.c_int)
        
        #temporary method for index awareness inside kernel.
        self._internal_index = data.ScalarArray(dtype=ctypes.c_int)
        self._internal_N = data.ScalarArray(dtype=ctypes.c_int)
        self._internal_index[0]=0
        self._internal_N[0] = self._N         
        
        self._cell_sort_code = '''
        
        const double R0 = P[0]+0.5*E[0];
        const double R1 = P[1]+0.5*E[1];
        const double R2 = P[2]+0.5*E[2];
        
        const int C0 = (int)(R0/CEL[0]);
        const int C1 = (int)(R1/CEL[1]);
        const int C2 = (int)(R2/CEL[2]);
        
        const int val = (C2*CA[1] + C1)*CA[0] + C0;
        
        Q[I[0]] = Q[N[0] + val];
        Q[N[0] + val] = I[0];
        I[0]++;
        
        '''
        self._cell_sort_dict = {'E':self._domain.extent,
                                'P':self._P,
                                'CEL':self._domain.cell_edge_lengths,
                                'CA':self._domain.cell_array,
                                'Q':self._q_list,
                                'I':self._internal_index,
                                'N':self._internal_N}
                
        
        
        self._cell_sort_kernel = kernel.Kernel('cell_list_method', self._cell_sort_code, headers = ['stdio.h'])
        self._cell_sort_loop = loop.SingleAllParticleLoop(self._N, self._cell_sort_kernel, self._cell_sort_dict)
        
    #move this to C    
    def _cell_sort_all(self):
        """
        Construct neighbour list, assigning atoms to cells. Using Rapaport algorithm.
        """

                
        for cx in range(self._domain.cell_count):
            self._q_list[self._N + cx] = -1
        
        
        self._internal_index[0]=0
        self._cell_sort_loop.execute()    
             
     
################################################################################################################
# DOUBLE PARTICLE LOOP
################################################################################################################        
        
class DoubleAllParticleLoop(loop.SingleAllParticleLoop):
    '''
    Class to loop over all particle pairs once.
    
    :arg int N: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg list headers: list containing C headers required by kernel.
    '''     
    def _compiler_set(self):
        self._cc = build.TMPCC        
    
    def _kernel_methodname(self):
        '''Construct the name of the kernel method.
        
        Return a string of the form 
        ``inline void kernel_name(double **<arg1>, double *<arg2}, ...) {``
        which is used for defining the name of the kernel method.
        '''
        space = ' '*14
        s = 'inline void '+self._kernel.name+'('
        
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            if (type(dat[1]) == particle.Dat):
                s += data.ctypes_map[dat[1].dtype]+' **'+dat[0]+', '
                
            if (type(dat[1]) == data.ScalarArray):
                s += data.ctypes_map[dat[1].dtype]+' *'+dat[0]+', '
            
            
        s = s[:-2] + ') {'
        return s

    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          for (int i=0; i<n; i++) { for (int j=0; j<i; j++) {  
              
              %(KERNEL_ARGUMENT_DECL)s
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
          }}
        }
        '''
    
    def _kernel_argument_declarations(self):
        '''Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a 
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as 
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        '''
        s = '\n'
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            
            space = ' '*14
            argname = dat[0]+'_ext'
            loc_argname = dat[0]
            
            
            if (type(dat[1]) == data.ScalarArray):
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+' = '+argname+';\n'
            
            if (type(dat[1]) == particle.Dat):
                ncomp = dat[1].ncomp
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+'[2];\n'
                s += space+loc_argname+'[0] = '+argname+'+'+str(ncomp)+'*i;\n'
                s += space+loc_argname+'[1] = '+argname+'+'+str(ncomp)+'*j;\n'    
        
        return s    
          
################################################################################################################
# DOUBLE ALL PARTICLE LOOP OPENMP
################################################################################################################        

class DoubleAllParticleLoopOpenMP(DoubleAllParticleLoop):
    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP    
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          
          #pragma omp parallel for schedule(dynamic)
          for (int i=0; i<n; i++) { 
            for (int j=0; j<i; j++) {  
              
              %(KERNEL_ARGUMENT_DECL)s
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
          }}
        }
        '''


################################################################################################################
# RAPAPORT LOOP OPENMP1
################################################################################################################

class PairLoopRapaportOpenMP1(PairLoopRapaport):
    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <omp.h>
        
        inline void cell_index_offset(const unsigned int cp, const unsigned int cpp_i, int* cell_array, double *d_extent, unsigned int* cpp, unsigned int *flag, double *offset){
        
            const int cell_map[14][3] = {   {0,0,0},
                                            {1,0,0},
                                            {0,1,0},
                                            {1,1,0},
                                            {1,-1,0},
                                            {-1,1,1},
                                            {0,1,1},
                                            {1,1,1},
                                            {-1,0,1},
                                            {0,0,1},
                                            {1,0,1},
                                            {-1,-1,1},
                                            {0,-1,1},
                                            {1,-1,1}};     
                
                
            unsigned int tmp = cell_array[0]*cell_array[1];    
            int Cz = cp/tmp;
            int Cx = cp %% cell_array[0];
            int Cy = (cp - Cz*tmp)/cell_array[0];
            
            
            Cx += cell_map[cpp_i][0];
            Cy += cell_map[cpp_i][1];
            Cz += cell_map[cpp_i][2];
            
            int C0 = (Cx + cell_array[0]) %% cell_array[0];    
            int C1 = (Cy + cell_array[1]) %% cell_array[1];
            int C2 = (Cz + cell_array[2]) %% cell_array[2];
                
             
            if ((Cx != C0) || (Cy != C1) || (Cz != C2)) { 
                *flag = 1;
                offset[0] = ((double)sign(Cx - C0))*d_extent[0];
                offset[1] = ((double)sign(Cy - C1))*d_extent[1];
                offset[2] = ((double)sign(Cz - C2))*d_extent[2];
                
                
                
            } else {*flag = 0; }
            
            *cpp = (C2*cell_array[1] + C1)*cell_array[0] + C0;
            
            
                
            return;      
        }    
        
        void %(KERNEL_NAME)s_wrapper(const int n, const int cell_count, int* cell_array, int* q_list, double* d_extent,%(ARGUMENTS)s) { 
            
            #pragma omp parallel for schedule(static, 100)
            for(unsigned int cp = 0; cp < cell_count; cp++){
                for(unsigned int cpp_i=0; cpp_i<14; cpp_i++){
                
                    double s[3]; 
                    unsigned int flag, cpp; 
                    int i,j;
                    
                    cell_index_offset(cp, cpp_i, cell_array, d_extent, &cpp, &flag, s);
                    
                    
                    double r1[3];
                    
                    i = q_list[n+cp];
                    while (i > -1){
                        j = q_list[n+cpp];
                        while (j > -1){
                            if (cp != cpp || i < j){
        
                                %(KERNEL_ARGUMENT_DECL)s
                                
                                  //KERNEL CODE START
                                  
                                  %(KERNEL)s
                                  
                                  //KERNEL CODE END
                                
                                
                            }
                            j = q_list[j];  
                        }
                        i=q_list[i];
                    }
                }
            }
            
            
            return;
        }        
        
        
        '''


################################################################################################################
# RAPAPORT LOOP OPENMP2
################################################################################################################

class PairLoopRapaportOpenMP2(PairLoopRapaport):
    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <omp.h>
        
        inline void cell_index_offset(const unsigned int cp, const unsigned int cpp_i, int* cell_array, double *d_extent, unsigned int* cpp, unsigned int *flag, double *offset){
        
            const int cell_map[27][3] = {   
                                            {-1,1,-1},
                                            {-1,-1,-1},
                                            {-1,0,-1},
                                            {0,1,-1},
                                            {0,-1,-1},
                                            {0,0,-1},
                                            {1,0,-1},
                                            {1,1,-1},
                                            {1,-1,-1},                                            

                                            {-1,1,0},
                                            {-1,0,0},                                            
                                            {-1,-1,0},
                                            {0,-1,0},
                                            {0,0,0},
                                            {0,1,0},
                                            {1,0,0},                                            
                                            {1,1,0},
                                            {1,-1,0},
                                            
                                            {-1,0,1},
                                            {-1,1,1},
                                            {-1,-1,1},
                                            {0,0,1},
                                            {0,1,1},
                                            {0,-1,1},
                                            {1,0,1},
                                            {1,1,1},
                                            {1,-1,1}
                                            
                                            };
                                            
                                             
                
                
            unsigned int tmp = cell_array[0]*cell_array[1];    
            int Cz = cp/tmp;
            int Cx = cp %% cell_array[0];
            int Cy = (cp - Cz*tmp)/cell_array[0];
            
            
            Cx += cell_map[cpp_i][0];
            Cy += cell_map[cpp_i][1];
            Cz += cell_map[cpp_i][2];
            
            int C0 = (Cx + cell_array[0]) %% cell_array[0];    
            int C1 = (Cy + cell_array[1]) %% cell_array[1];
            int C2 = (Cz + cell_array[2]) %% cell_array[2];
                
             
            if ((Cx != C0) || (Cy != C1) || (Cz != C2)) { 
                *flag = 1;
                offset[0] = ((double)sign(Cx - C0))*d_extent[0];
                offset[1] = ((double)sign(Cy - C1))*d_extent[1];
                offset[2] = ((double)sign(Cz - C2))*d_extent[2];
                
                
                
            } else {*flag = 0; }
            
            *cpp = (C2*cell_array[1] + C1)*cell_array[0] + C0;
            
            
                
            return;      
        }    
        
        void %(KERNEL_NAME)s_wrapper(const int n, const int cell_count, int* cell_array, int* q_list, double* d_extent,%(ARGUMENTS)s) { 
            
            %(OPENMP_INIT)s
            
            #pragma omp parallel for schedule(dynamic) %(OPENMP_DECLARATION)s
            for(unsigned int cp = 0; cp < cell_count; cp++){
                for(unsigned int cpp_i=0; cpp_i<27; cpp_i++){
                
                    double s[3]; 
                    unsigned int flag, cpp; 
                    int i,j;
                    
                    cell_index_offset(cp, cpp_i, cell_array, d_extent, &cpp, &flag, s);
                    
                    
                    double r1[3];
                    
                    i = q_list[n+cp];
                    while (i > -1){
                        j = q_list[n+cpp];
                        while (j > -1){
                            if ((cp != cpp) || (i != j)){
        
                                %(KERNEL_ARGUMENT_DECL)s
                                
                                  //KERNEL CODE START
                                  
                                  %(KERNEL)s
                                  
                                  //KERNEL CODE END
                                
                                
                            }
                            j = q_list[j];  
                        }
                        i=q_list[i];
                    }
                }
            }
            
            %(OPENMP_FINALISE)s
            
            return;
        }        
        
        
        '''

    def _generate_impl_source(self):
        '''Generate the source code the actual implementation.
        '''

        d = {'UNIQUENAME':self._unique_name,
             'KERNEL_METHODNAME':self._kernel_methodname(),
             'KERNEL':self._kernel.code,
             'ARGUMENTS':self._argnames(),
             'LOC_ARGUMENTS':self._loc_argnames(),
             'KERNEL_NAME':self._kernel.name,
             'KERNEL_ARGUMENT_DECL':self._kernel_argument_declarations(),
             'OPENMP_INIT':self._kernel.OpenMPInitStr,
             'OPENMP_DECLARATION':self._kernel.OpenMPDecStr,
             'OPENMP_FINALISE':self._kernel.OpenMPFinalStr}
        return self._code % d        
    
    
    
    
    
    
    
    
    
    
    
