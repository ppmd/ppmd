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



################################################################################################################
# SINGLE PARTICLE LOOP SERIAL
################################################################################################################
class SingleAllParticleLoop(object):
    '''
    Class to loop over all particles once.
    
    :arg int N: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    '''
    def __init__(self, N, kernel, particle_dat_dict):
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
        self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)

     
        

    def _create_library(self):
        '''
        Create a shared library from the source code.
        '''
        
        filename_base = os.path.join(self._temp_dir,self._unique_name)
        header_filename = filename_base+'.h'
        impl_filename = filename_base+'.c'
        with open(header_filename,'w') as f:
            print >> f, self._generate_header_source()        
        with open(impl_filename,'w') as f:
            print >> f, self._generate_impl_source()
        object_filename = filename_base+'.o'
        library_filename = filename_base+'.so'        
        cflags = ['-O3','-fpic','-std=c99','-lm']
        cc = 'gcc'
        ld = 'gcc'
        link_flags = ['-lm']
        compile_cmd = [cc,'-c','-fpic']+cflags+['-I',self._temp_dir] \
                       +['-o',object_filename,impl_filename]
        link_cmd = [ld,'-shared']+link_flags+['-o',library_filename,object_filename]
        stdout_filename = filename_base+'.log'
        stderr_filename = filename_base+'.err'
        with open(stdout_filename,'w') as stdout:
            with open(stderr_filename,'w') as stderr:
                stdout.write('Compilation command:\n')
                stdout.write(' '.join(compile_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(compile_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()
                stdout.write('Link command:\n')
                stdout.write(' '.join(link_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(link_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()                
        
    def execute(self, dat_dict = None):
    
        if (dat_dict != None):
            self._particle_dat_dict = dat_dict    
    
        '''Execute the kernel over all particle pairs.'''
        args = []
        for dat in self._particle_dat_dict.values():
            args.append(dat.ctypes_data)
            
        method = self._lib[self._kernel.name+'_wrapper']
        method(self._N,*args)   
    

       
    def _generate_impl_source(self):
        '''Generate the source code the actual implementation.
        '''

        d = {'UNIQUENAME':self._unique_name,
             'KERNEL_METHODNAME':self._kernel_methodname(),
             'KERNEL':self._kernel.code,
             'ARGUMENTS':self._argnames(),
             'LOC_ARGUMENTS':self._loc_argnames(),
             'KERNEL_NAME':self._kernel.name,
             'KERNEL_ARGUMENT_DECL':self._kernel_argument_declarations()}
        return self._code % d        
        
    
    def _kernel_methodname(self):
        '''Construct the name of the kernel method.
        
        Return a string of the form 
        ``inline void kernel_name(double *<arg1>, double *<arg2}, ...) {``
        which is used for defining the name of the kernel method.
        '''
        space = ' '*14
        s = 'inline void '+self._kernel.name+'('
        for var_name_kernel, var_name_state  in self._particle_dat_dict.items():
            #print var_name_kernel, var_name_state.dattype()
            s += data.ctypes_map[var_name_state.dtype]+' *'+var_name_kernel+', '
     
        
        
        s = s[:-2] + ') {'
        return s

    def _loc_argnames(self):
        '''Comma separated string of local argument names.

        This string is used in the call to the local kernel. If, for
        example, two particle dats get passed to the pairloop, then
        the result will be ``loc_arg_000,loc_arg_001``. Each of these
        is of type ``double* [2]``, see method _kernel_argument_declarations()
        '''
        argnames = ''
        for i in range(self._nargs):
            argnames += 'loc_arg_'+('%03d' % i)+','
        return argnames[:-1]

    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        %(KERNEL_METHODNAME)s
        %(KERNEL)s
        }

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          int i;
          for (i=0; i<n; ++i) {
              %(KERNEL_ARGUMENT_DECL)s
              %(KERNEL_NAME)s(%(LOC_ARGUMENTS)s);
              
            }
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
            argname = 'arg_'+('%03d' % i)
            loc_argname = 'loc_'+argname
            
            
            
            if (type(dat[1]) == data.ScalarArray):
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+' = '+argname+';\n'
            
            if (type(dat[1]) == particle.Dat):
                
                ncomp = dat[1].ncomp
                s += space+data.ctypes_map[dat[1].dtype]+' *'+loc_argname+';\n'
                s += space+loc_argname+' = '+argname+'+'+str(ncomp)+'*i;\n'     
        
        return s 

    """   
    def _argnames(self):
        '''Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of 
        the method which executes the pairloop over the grid. 
        If, for example, the pairloop gets passed two particle_dats, 
        then the result will be ``double* arg_000,double* arg_001`.`
        '''
        
        argnames = ''
        for i in range(self._nargs):
            argnames += 'double *arg_'+('%03d' % i)+','
        return argnames[:-1]        
     """
      
    def _argnames(self):
        '''Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of 
        the method which executes the pairloop over the grid. 
        If, for example, the pairloop gets passed two particle_dats, 
        then the result will be ``double** arg_000,double** arg_001`.`
        '''
        #argnames = ''
        #for i in range(self._nargs):
        #    argnames += 'double **arg_'+('%03d' % i)+','
        #return argnames[:-1]
        
        argnames = ''
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            argnames += data.ctypes_map[dat[1].dtype]+' *arg_'+('%03d' % i)+','
            

        return argnames[:-1]        
        
        
        
    def _generate_header_source(self):
        '''Generate the source code of the header file.

        Returns the source code for the header file.
        '''
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        
        #include "../generic.h"
        
        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME':self._unique_name,
             'INCLUDED_HEADERS':self._included_headers(),
             'KERNEL_NAME':self._kernel.name,
             'ARGUMENTS':self._argnames()}
        return (code % d)

    def _included_headers(self):
        '''Return names of included header files.'''
        s = ''
        if (self._kernel.headers != None):
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"'+x+'\" \n'
        return s
        

    def _unique_name_calc(self):
        '''Return name which can be used to identify the pair loop 
        in a unique way.
        '''
        return self._kernel.name+'_'+self.hexdigest()
        
    def hexdigest(self):
        '''Create unique hex digest'''
        m = hashlib.md5()
        m.update(self._kernel.code+self._code)
        if (self._kernel.headers != None):
            for header in self._kernel.headers:
                m.update(header)
        return m.hexdigest()

################################################################################################################
# SINGLE PARTICLE LOOP OPENMP
################################################################################################################

class SingleAllParticleLoopOpenMP(SingleAllParticleLoop):
    '''
    OpenMP version of single pass pair loop (experimental)
    '''
    SingleAllParticleLoop._create_library
    
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <omp.h>

        #include "../generic.h"

        %(KERNEL_METHODNAME)s
        %(KERNEL)s
        }

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          int i;
          #pragma omp parallel for schedule(static)
          for (i=0; i<n; ++i) {
              %(KERNEL_ARGUMENT_DECL)s
              %(KERNEL_NAME)s(%(LOC_ARGUMENTS)s);
              
            }
        }
        '''
    
          
    
    def _create_library(self):
        '''
        Create a shared library from the source code.
        '''
        
        filename_base = os.path.join(self._temp_dir,self._unique_name)
        header_filename = filename_base+'.h'
        impl_filename = filename_base+'.c'
        with open(header_filename,'w') as f:
            print >> f, self._generate_header_source()        
        with open(impl_filename,'w') as f:
            print >> f, self._generate_impl_source()
        object_filename = filename_base+'.o'
        library_filename = filename_base+'.so'        
        cflags = ['-O3','-fpic','-fopenmp','-lgomp','-lpthread','-lc','-lrt']
        cc = 'gcc'
        ld = 'gcc'
        link_flags = ['-fopenmp','-lgomp','-lpthread','-lc','-lrt']
        compile_cmd = [cc,'-c','-fpic']+cflags+['-I',self._temp_dir] \
                       +['-o',object_filename,impl_filename]
        link_cmd = [ld,'-shared']+link_flags+['-o',library_filename,object_filename]
        stdout_filename = filename_base+'.log'
        stderr_filename = filename_base+'.err'
        with open(stdout_filename,'w') as stdout:
            with open(stderr_filename,'w') as stderr:
                stdout.write('Compilation command:\n')
                stdout.write(' '.join(compile_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(compile_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()
                stdout.write('Link command:\n')
                stdout.write(' '.join(link_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(link_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate() 
        

