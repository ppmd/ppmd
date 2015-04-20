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
import build



class _base(build.GenericToolChain):
    '''
    Generic base class to loop over all particles once.
    
    :arg int N: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg bool DEBUG: Flag to enable debug flags.
    '''
    def __init__(self, N, kernel, particle_dat_dict, DEBUG = False):
        self._DEBUG = DEBUG
        self._compiler_set()
        self._N = N
        self._temp_dir = './build/'
        if (not os.path.exists(self._temp_dir)):
            os.mkdir(self._temp_dir)
        self._kernel = kernel
        
        self._particle_dat_dict = particle_dat_dict
        self._nargs = len(self._particle_dat_dict)

        self._code_init()
        
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

    def _kernel_methodname(self):
        '''Construct the name of the kernel method.
        
        Return a string of the form 
        ``inline void kernel_name(double *<arg1>, double *<arg2}, ...) {``
        which is used for defining the name of the kernel method.
        '''
        space = ' '*14
        s = 'inline void '+self._kernel.name+'('
        
        #for var_name_kernel, var_name_state  in self._particle_dat_dict.items():
        for i,dat in enumerate(self._particle_dat_dict.items()):
            #print var_name_kernel, var_name_state.dattype()
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
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+';\n'
                s += space+loc_argname+' = '+argname+'+'+str(ncomp)+'*i;\n'     
        
        
        return s
    
    def execute(self, dat_dict = None):
    
        if (dat_dict != None):
            self._particle_dat_dict = dat_dict    
    
        '''Execute the kernel over all particle pairs.'''
        args = []
        for dat in self._particle_dat_dict.values():
            args.append(dat.ctypes_data)
            
        method = self._lib[self._kernel.name+'_wrapper']
        method(self._N,*args)           
              
################################################################################################################
# SINGLE PARTICLE LOOP SERIAL
################################################################################################################
class SingleAllParticleLoop(_base):
                   
    def _compiler_set(self):
        self._cc = build.TMPCC
        
        
    
    def _code_init(self):
        self._kernel_code = self._kernel.code
    
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          int i;
          for (i=0; i<n; ++i) {
              %(KERNEL_ARGUMENT_DECL)s
              
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
            }
        }
        '''

################################################################################################################
# SINGLE PARTICLE LOOP OPENMP
################################################################################################################

class SingleAllParticleLoopOpenMP(SingleAllParticleLoop):
    '''
    OpenMP version of single pass pair loop (experimental)
    '''
    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP
    
    def _code_init(self):
        self._kernel_code = self._kernel.code
    
        self._ompinitstr = ''
        self._ompdecstr = ''
        self._ompfinalstr = '' 
    
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <omp.h>

        #include "../generic.h"

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          int i;
          
          %(OPENMP_INIT)s
          
          #pragma omp parallel for schedule(dynamic) %(OPENMP_DECLARATION)s
          for (i=0; i<n; ++i) {
              %(KERNEL_ARGUMENT_DECL)s
              
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
            }
            
            %(OPENMP_FINALISE)s
            
        }
        
        '''
    
    def _generate_impl_source(self):
        '''Generate the source code the actual implementation.
        '''
        

        d = {'KERNEL_ARGUMENT_DECL':self._kernel_argument_declarations_openmp(),
             'UNIQUENAME':self._unique_name,
             'KERNEL':self._kernel_code,
             'ARGUMENTS':self._argnames(),
             'LOC_ARGUMENTS':self._loc_argnames(),
             'KERNEL_NAME':self._kernel.name,
             'OPENMP_INIT':self._ompinitstr,
             'OPENMP_DECLARATION':self._ompdecstr,
             'OPENMP_FINALISE':self._ompfinalstr
             }
             
        
        return self._code % d     
        
    def _kernel_argument_declarations_openmp(self):
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

            reduction_handle = self._kernel.reduction_variable_lookup(dat[0])
            
            if (reduction_handle != None):
                if (dat[1].ncomp != 1): 
                    print "WARNING, Reductions not valid for more than 1 element"
                
                #Create a var name a variable to reduce upon.
                reduction_argname = dat[0]+'_reduction'
                
                #Initialise variable
                self._ompinitstr += data.ctypes_map[dat[1].dtype]+' '+reduction_argname+' = '+build.omp_operator_init_values[reduction_handle.operator]+';'
                
                #Add to omp pragma
                self._ompdecstr += 'reduction('+reduction_handle.operator+':'+reduction_argname+')'
                
                #Modify kernel code to use new reduction variable.
                self._kernel_code = build.replace(self._kernel_code,reduction_handle.pointer, reduction_argname)
                
                #write final value to output pointer
                
                self._ompfinalstr += argname+'['+reduction_handle.index+'] ='+reduction_argname+';'
                
            
            else:
            
                if (type(dat[1]) == data.ScalarArray):
                    s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+' = '+argname+';\n'
                
                elif (type(dat[1]) == particle.Dat):
                    
                    ncomp = dat[1].ncomp
                    s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+';\n'
                    s += space+loc_argname+' = '+argname+'+'+str(ncomp)+'*i;\n'     
        
        
        return s
