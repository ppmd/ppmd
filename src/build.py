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
import constant


class compiler(object):
    '''
    Container to define different compilers.
    
    :arg str name: Compiler name, referance only.
    :arg str binary: Name(+path) of compiler binary.
    :arg list cflags: List of compile flags as strings.
    :arg list lflags: List of link flags as strings.
    :arg list dbgflags: List of debug flags as strings.
    :arg list compileflag: List of compile flag as single string (eg ['-c'] for gcc).
    :arg list sharedlibraryflag: List of flags as strings to link as shared library.
    '''
    def __init__(self,name,binary,cflags,lflags,dbgflags,compileflag,sharedlibraryflag):
        self._name = name
        self._binary = binary
        self._cflags = cflags
        self._lflags = lflags
        self._dbgflags = dbgflags
        self._compileflag = compileflag
        self._sharedlibf = sharedlibraryflag
    @property    
    def name(self):
        '''Return compiler name.'''
        return self._name
    @property
    def binary(self):
        '''Return compiler binary.'''
        return self._binary
    @property
    def cflags(self):
        '''Return compiler compile flags'''
        return self._cflags
    @property
    def lflags(self):
        '''Return compiler link flags'''
        return self._lflags
    @property
    def dbgflags(self):
        '''Return compiler debug flags'''
        return self._dbgflags
    @property
    def compileflag(self):
        '''Return compiler compile flag.'''
        return self._compileflag
    @property
    def sharedlibflag(self):
        '''Return compiler link as shared library flag.'''
        return self._sharedlibf            


#Define system gcc version as compiler.
GCC = compiler(['GCC'],['gcc'],['-O3','-fpic','-std=c99'],['-lm'],['-g'],['-c'],['-shared'])

#Define system gcc version as OpenMP compiler.
GCC_OpenMP = compiler(['GCC'],['gcc'],['-O3','-fpic','-fopenmp','-lgomp','-lpthread','-lc','-lrt','-std=c99'],['-fopenmp','-lgomp','-lpthread','-lc','-lrt'],['-g'],['-c'],['-shared'])

#Define system icc version as compiler.
ICC = compiler(['ICC'],['icc'],['-O3','-fpic','-std=c99','-march=core-avx2'],['-lm'],['-g'],['-c'],['-shared'])

#Define system icc version as OpenMP compiler.
ICC_OpenMP = compiler(['ICC'],['icc'],['-O3','-fpic','-openmp','-lgomp','-lpthread','-lc','-lrt','-std=c99','-march=core-avx2'],['-openmp','-lgomp','-lpthread','-lc','-lrt'],['-g'],['-c'],['-shared'])





class GenericToolChain(object):
    def _argnames(self):
        '''Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of 
        the method which executes the pairloop over the grid. 
        If, for example, the pairloop gets passed two particle_dats, 
        then the result will be ``double** arg_000,double** arg_001`.`
        '''
        
        argnames = ''
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            argnames += data.ctypes_map[dat[1].dtype]+' *'+dat[0]+'_ext,'
            
        return argnames[:-1]  
        
    def _loc_argnames(self):
        '''Comma separated string of local argument names.
        '''
        argnames = ''
        for i,dat in enumerate(self._particle_dat_dict.items()):
            argnames += dat[0]+','
        return argnames[:-1]        

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


    def _included_headers(self):
        '''Return names of included header files.'''
        s = ''
        if (self._kernel.headers != None):
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"'+x+'\" \n'
        return s

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
        cflags = self._cc.cflags
        if (self._DEBUG):
            cflags+=self._cc.dbgflags
        cc = self._cc.binary
        ld = self._cc.binary
        lflags = self._cc.lflags
        compile_cmd = cc+self._cc.compileflag+cflags+['-I',self._temp_dir] \
                       +['-o',object_filename,impl_filename]
        link_cmd = ld+self._cc.sharedlibflag+lflags+['-o',library_filename,object_filename]
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










