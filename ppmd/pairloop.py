# system level
import numpy as np
import ctypes
import os
import hashlib
import subprocess

# package level
import generation
import data
import build
import runtime
import access
import cell
import host
import opt


class _Base(object):

    def __init__(self, n, types_map, kernel, particle_dat_dict):

        self._cc = build.TMPCC

        self._N = n
        self._types_map = types_map

        self._kernel = kernel

        self._particle_dat_dict = particle_dat_dict

        self.loop_timer = opt.LoopTimer()

        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)

    def _kernel_argument_declarations(self):
        """Define and declare the kernel arguments.

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
        """
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            space = ' ' * 14

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            argname = dat[0] + '_ext'
            loc_argname = dat[0]

            if type(dat[1]) == data.ScalarArray:
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == data.ParticleDat:
                ncomp = dat[1].ncomp
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == data.TypedDat:

                ncomp = dat[1].ncomp
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

    def _argnames(self):
        """Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of
        the method which executes the pairloop over the grid.
        If, for example, the pairloop gets passed two particle_dats,
        then the result will be ``double** arg_000,double** arg_001`.`
        """


        argnames = str(self.loop_timer.get_cpp_arguments()) + ','


        if self._kernel.static_args is not None:
            self._static_arg_order = []

            for i, dat in enumerate(self._kernel.static_args.items()):
                argnames += 'const ' + host.ctypes_map[dat[1]] + ' ' + dat[0] + ','
                self._static_arg_order.append(dat[0])


        for i, dat in enumerate(self._particle_dat_dict.items()):
            if type(dat[1]) is not tuple:
                argnames += host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'
            else:
                argnames += host.ctypes_map[dat[1][0].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'


        return argnames[:-1]

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + x + '\" \n'

        s += str(self.loop_timer.get_cpp_headers())

        return s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations(),
             'LOOP_TIMER_PRE': str(self.loop_timer.get_cpp_pre_loop_code()),
             'LOOP_TIMER_POST': str(self.loop_timer.get_cpp_post_loop_code())}

        return self._code % d

    def execute(self, n=None, dat_dict=None, static_args=None):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Currently assume n is always needed'''
        if n is not None:
            _N = n
        else:
            _N = self._N()

        args = [ctypes.c_int(_N)]

        if self._types_map is not None:
            args.append(self._types_map.ctypes_data)


        args.append(self.loop_timer.get_python_parameters())


        '''TODO IMPLEMENT/CHECK RESISTANCE TO ARG REORDERING'''

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']
        method(*args)

        '''after wards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])



class DoubleAllParticleLoop(_Base):
    """
    Class to loop over all particle pairs once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg list headers: list containing C headers required by kernel.
    """

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        extern "C" void %(KERNEL_NAME)s_wrapper(int n, int *_TYPE_MAP,%(ARGUMENTS)s);

        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''

        void %(KERNEL_NAME)s_wrapper(const int n, int *_TYPE_MAP,%(ARGUMENTS)s) { 
          for (int i=0; i<n; i++) { for (int j=0; j<i; j++) {  
              
              %(KERNEL_ARGUMENT_DECL)s
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
          }}
        }
        '''

    def _kernel_argument_declarations(self):
        """Define and declare the kernel arguments.

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
        """
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            space = ' ' * 14
            argname = dat[0] + '_ext'
            loc_argname = dat[0]

            if type(dat[1]) == data.ScalarArray:
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == data.ParticleDat:
                ncomp = dat[1].ncomp
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == data.TypedDat:

                ncomp = dat[1].ncomp
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

###############################################################################
# DOUBLE PARTICLE LOOP APPLIES PBC
###############################################################################


class DoubleAllParticleLoopPBC(DoubleAllParticleLoop):
    """
    Generic base class to loop over all particles once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, n, domain, kernel, particle_dat_dict):

        self._compiler_set()
        self._N = n
        self._domain = domain
        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)
        self._kernel = kernel

        self._particle_dat_dict = particle_dat_dict

        self.loop_timer = opt.LoopTimer()

        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''

        void %(KERNEL_NAME)s_wrapper(const int n,double *extent_ext ,%(ARGUMENTS)s) {
          
          const double _E_2[3] = {0.5*extent_ext[0], 0.5*extent_ext[1], 0.5*extent_ext[2]};
          const double _E[3] = {extent_ext[0], extent_ext[1], extent_ext[2]};          
           
          for (int i=0; i<n; i++) { for (int j=0; j<i; j++) {  
              
              double r1[3], s1[3];

              
              %(KERNEL_ARGUMENT_DECL)s
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
          }}
        }
        '''

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        extern "C" void %(KERNEL_NAME)s_wrapper(int n,double *extent_ext,%(ARGUMENTS)s);

        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _kernel_argument_declarations(self):
        """Define and declare the kernel arguments.

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
        """
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW


            space = ' ' * 14
            argname = dat[0] + '_ext'
            loc_argname = dat[0]

            if type(dat[1]) == data.ScalarArray:
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'positions':
                    s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

                    s += space + 'r1[0] =' + argname + '[LINIDX_2D(3,i,0)] -' + argname + '[LINIDX_2D(3,j,0)]; \n'
                    s += space + 'r1[1] =' + argname + '[LINIDX_2D(3,i,1)] -' + argname + '[LINIDX_2D(3,j,1)]; \n'
                    s += space + 'r1[2] =' + argname + '[LINIDX_2D(3,i,2)] -' + argname + '[LINIDX_2D(3,j,2)]; \n'


                    # s += space+'s1[0] = ((abs_md(r1[0]))>_E_2[0]? sign(r1[0])*_E[0]:0.0); \n'
                    # s += space+'s1[1] = ((abs_md(r1[1]))>_E_2[1]? sign(r1[1])*_E[1]:0.0); \n'
                    # s += space+'s1[2] = ((abs_md(r1[2]))>_E_2[2]? sign(r1[2])*_E[2]:0.0); \n'

                    s += space + 's1[0] = r1[0]>_E_2[0] ? _E[0]:  r1[0]<-1*_E_2[0] ? -1*_E[0]:  0 ; \n'
                    s += space + 's1[1] = r1[1]>_E_2[1] ? _E[1]:  r1[1]<-1*_E_2[1] ? -1*_E[1]:  0 ; \n'
                    s += space + 's1[2] = r1[2]>_E_2[2] ? _E[2]:  r1[2]<-1*_E_2[2] ? -1*_E[2]:  0 ; \n'

                    s += space + loc_argname + '[0] = r1;\n'
                    s += space + loc_argname + '[1] = s1;\n'


                else:
                    ncomp = dat[1].ncomp
                    s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                    s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == data.TypedDat:

                ncomp = dat[1].ncomp
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

    def execute(self, dat_dict=None, static_args=None):

        # Allow alternative pointers
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Currently assume n is always needed'''
        args = [self._N(), self._domain.extent.ctypes_data]

        args.append(self.loop_timer.get_python_parameters())

        '''TODO IMPLEMENT/CHECK RESISTANCE TO ARG REORDERING'''

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']
        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


###############################################################################
# RAPAPORT LOOP SERIAL FOR HALO DOMAINS
###############################################################################


class PairLoopRapaportHalo(_Base):
    def __init__(self, domain, potential=None, dat_dict=None, kernel=None):
        self._domain = domain
        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        self.loop_timer = opt.LoopTimer()

        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)

    def _kernel_argument_declarations(self):
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        
        extern "C" void %(KERNEL_NAME)s_wrapper(const int n, int* cell_array, int* q_list,%(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include <stdio.h>
        
        void cell_index_offset(const unsigned int cp_i, const unsigned int cpp_i, int* cell_array, unsigned int* cpp, unsigned int* cp_h_flag, unsigned int* cpp_h_flag){
        
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
            
            //internal cell array dimensions
            const unsigned int ca0 = cell_array[0];
            const unsigned int ca1 = cell_array[1];
            const unsigned int ca2 = cell_array[2];
            
            //Get index for cell cp as tuple: cp=(Cx, Cy, Cz)
            const int tmp = ca0*ca1;
            int Cz = cp_i/tmp;
            int Cx = cp_i %% ca0;
            int Cy = (cp_i - Cz*tmp)/ca0;
            
            //check if cp is in halo
            if( (Cx %% (ca0-1) == 0) || (Cy %% (ca1-1) == 0) || (Cz %% (ca2-1) == 0)){
                *cp_h_flag = 1;
            }else{*cp_h_flag = 0;}
            
            
            //compute tuple of adjacent cell cpp
            Cx = (Cx + cell_map[cpp_i][0] + ca0) %% ca0;
            Cy = (Cy + cell_map[cpp_i][1] + ca1) %% ca1;
            Cz = (Cz + cell_map[cpp_i][2] + ca2) %% ca2;
            
            //compute linear index of adjacent cell cpp
            *cpp = Cz*(ca0*ca1) + Cy*(ca0) + Cx;
            
            //check if cpp is in halo
            if( (Cx %% (ca0-1) == 0) || (Cy %% (ca1-1) == 0) || (Cz %% (ca2-1) == 0)){
                *cpp_h_flag = 1;
            }else{*cpp_h_flag = 0;}          
            
            
                
            return;      
        }    
        
        void %(KERNEL_NAME)s_wrapper(const int n, int* cell_array, int* q_list,%(ARGUMENTS)s) { 
            
            //printf("starting");
            for(unsigned int cp = 0; cp < cell_array[0]*cell_array[1]*(cell_array[2]-1); cp++){
                for(unsigned int cpp_i=0; cpp_i<14; cpp_i++){
                    
                    //printf("cp=%%d, cpp_i=%%d |",cp,cpp_i);
                    
                    
                    unsigned int cpp, _cp_halo_flag, _cpp_halo_flag;
                    int _i,_j;
                    
                    
                    cell_index_offset(cp, cpp_i, cell_array, &cpp, &_cp_halo_flag, &_cpp_halo_flag);
                    
                    //Check that both cells are not halo cells.
                    
                    //printf("cpp=%%d, flagi=%%d, flagj=%%d |",cpp, cp_h_flag,cpp_h_flag);
                    
                    if ((_cp_halo_flag+_cpp_halo_flag) < 2){
                        
                        _i = q_list[n+cp];
                        while (_i > -1){
                            _j = q_list[n+cpp];
                            while (_j > -1){
                                if (cp != cpp || _i < _j){
                                    
                                    double *ri, *rj;
                                    
                                    double null_array[3] = {0,0,0};
                                    //printf("i=%%d, j=%%d |",i,j);
                                    
                                    
                                    %(KERNEL_ARGUMENT_DECL)s
                                    
                                      //KERNEL CODE START
                                      
                                      %(KERNEL)s
                                      
                                      //KERNEL CODE END
                                    
                                    
                                }
                                _j = q_list[_j];
                            }
                            _i=q_list[_i];
                        }

                    }
                }
            }
            
            
            return;
        }        
        
        
        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """
        cell.cell_list.check()
        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Create arg list'''
        _halo_exchange_particle_dat(self._particle_dat_dict)
        if n is not None:
            print "warning option depreciated"
            #_N = n
        else:
            _N = cell.cell_list.cell_list[cell.cell_list.cell_list.end]

        args = [ctypes.c_int(_N),
                self._domain.cell_array.ctypes_data,
                cell.cell_list.cell_list.ctypes_data]

        args.append(self.loop_timer.get_python_parameters())


        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                #print dat_orig[0].name, dat_orig[1]
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                #print dat_orig.name, "else"
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


###############################################################################
# Neighbour list looping using NIII
###############################################################################

class PairLoopNeighbourList(_Base):
    def __init__(self, potential=None, dat_dict=None, kernel=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        self.loop_timer = opt.LoopTimer()

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)


        self.neighbour_list = cell.NeighbourList()
        self.neighbour_list.setup(*cell.cell_list.get_setup_parameters())







        self._neighbourlist_count = 0
        self._invocations = 0

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        extern "C" void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* START_POINTS, const int* NLIST, %(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _kernel_argument_declarations(self):
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return s

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include <stdio.h>

        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* __restrict__ START_POINTS, const int* __restrict__ NLIST, %(ARGUMENTS)s) {


            %(LOOP_TIMER_PRE)s


            for(int _i = 0; _i < N_LOCAL; _i++){
                for(int _k = START_POINTS[_i]; _k < START_POINTS[_i+1]; _k++){
                    int _j = NLIST[_k];
                    int _cpp_halo_flag;
                    int _cp_halo_flag = 0;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    //if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }
            }

            %(LOOP_TIMER_POST)s

            return;
        }


        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        cell.cell_list.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict



        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Rebuild neighbour list potentially'''
        self._invocations += 1
        if cell.cell_list.version_id > self.neighbour_list.version_id:
            self.neighbour_list.update()
            self._neighbourlist_count += 1

        '''Create arg list'''
        _N_TOTAL = ctypes.c_int(self.neighbour_list.n_total)
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_TOTAL,
                 _N_LOCAL,
                 _STARTS,
                 _LIST]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


###############################################################################
# Neighbour list with a list for inside and outside the halo
###############################################################################


class PairLoopNeighbourListHaloAware(_Base):
    def __init__(self, potential=None, dat_dict=None, kernel=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        self.loop_timer = opt.LoopTimer()

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)


        self.neighbour_list = cell.NeighbourListHaloAware()
        self.neighbour_list.setup(*cell.cell_list.get_setup_parameters())
        self._neighbourlist_count = 0
        self._invocations = 0


    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        extern "C" void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int* __restrict__ START_POINTS, const int* __restrict__ NLIST, %(ARGUMENTS)s);
        extern "C" void %(KERNEL_NAME)s_halo_wrapper(const int N_PART, const int N_LOCAL, const int* __restrict__ START_POINTS, const int* __restrict__ NLIST, %(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _kernel_argument_declarations(self):
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return s

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include <stdio.h>

        void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int* __restrict__ START_POINTS, const int* __restrict__ NLIST, %(ARGUMENTS)s) {

            for(int _i = 0; _i < N_LOCAL; _i++){
                for(int _k = START_POINTS[_i]; _k < START_POINTS[_i+1]; _k++){
                    int _j = NLIST[_k];
                    int _cpp_halo_flag = 0;
                    int _cp_halo_flag = 0;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    //if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                    //if (_j >= N_LOCAL) { _cpp_halo_flag = 1; printf("oh dear NLOCAL %%d, _j %%d \\n", N_LOCAL, _j); } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }
            }

            return;
        }

        void %(KERNEL_NAME)s_halo_wrapper(const int N_PART, const int N_LOCAL, const int* __restrict__ START_POINTS, const int* __restrict__ NLIST, %(ARGUMENTS)s) {

            for(int _ip = 0; _ip < N_PART; _ip++){
                for(int _k = START_POINTS[_ip*2]; _k < START_POINTS[(_ip+1)*2]; _k++){
                    int _i = START_POINTS[(_ip*2)+1];
                    int _j = NLIST[_k];
                    int _cpp_halo_flag = 1;
                    int _cp_halo_flag = 0;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    //if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                    //if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }
            }

            return;
        }

        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        cell.cell_list.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict


        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Rebuild neighbour list potentially'''
        self._invocations += 1
        if cell.cell_list.version_id > self.neighbour_list.version_id:

            self.neighbour_list.update()
            self.neighbour_list.halo_update()

            self._neighbourlist_count += 1

        '''Create arg list'''
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_LOCAL,
                 _STARTS,
                 _LIST]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

        args[1] = self.neighbour_list.halo_neighbour_starting_points.ctypes_data
        args[2] = self.neighbour_list.halo_list.ctypes_data

        args = [ctypes.c_int(self.neighbour_list.halo_part_count[0])] + args

        self._lib[self._kernel.name + '_halo_wrapper'](*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()




###############################################################################
# Neighbour list looping using NIII and Blocking
###############################################################################

class BlockPairLoopNeighbourList(_Base):
    def __init__(self, potential=None, dat_dict=None, kernel=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        self.loop_timer = opt.LoopTimer()

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)


        self.neighbour_list = cell.NeighbourList()
        self.neighbour_list.setup(*cell.cell_list.get_setup_parameters())







        self._neighbourlist_count = 0
        self._invocations = 0

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        extern "C" void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* START_POINTS, const int* NLIST, %(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _kernel_argument_declarations(self):
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return s

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include <stdio.h>

        #define _PBLOCK 8
        #define _NBLOCK 8


        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* __restrict__ START_POINTS, const int* __restrict__ NLIST, %(ARGUMENTS)s) {


            %(LOOP_TIMER_PRE)s

            const int _DIV = N_LOCAL/_PBLOCK;


            for(int _pbx = 0; _pbx < _DIV*_PBLOCK; _pbx+=_PBLOCK){
                //_pbx=0 , 0+pblock, 0+2*pblock....

                //find how deep we need to go in the neighbour list.
                int _NMIN = START_POINTS[_pbx+1] - START_POINTS[_pbx];
                for(int _tmpx = _pbx+1; _tmpx < _pbx+_PBLOCK; _tmpx++){
                    _NMIN = MIN(_NMIN, START_POINTS[_tmpx+1] - START_POINTS[_tmpx]);
                }

                //printf("NMIN=%%d \\n", _NMIN);

                // get number of blocks that can be looped over with no branching.
                const int _NNBLOCKS = _NMIN/_NBLOCK;

                // loop over blocks.


                for(int _tmpx = 0; _tmpx < _NNBLOCKS; _tmpx++){

                    for(int _i = _pbx; _i < _pbx + _PBLOCK ; _i++){


                        //loop over neighbours in block;
                        const int _nstart = START_POINTS[_i] + _tmpx*_NBLOCK;
                        for(int _k = _nstart; _k < _nstart+_NBLOCK; _k++){

                            int _j = NLIST[_k];

                            int _cpp_halo_flag;
                            int _cp_halo_flag = 0;

                            if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                             %(KERNEL_ARGUMENT_DECL)s

                             //KERNEL CODE START

                             %(KERNEL)s

                             //KERNEL CODE END


                        }
                    }

                }

                for(int _i = _pbx; _i < _pbx + _PBLOCK ; _i++){

                    //trailing ends of neighbours.
                    const int _nstart = START_POINTS[_i] + _NNBLOCKS*_NBLOCK;
                    for(int _k = _nstart; _k < START_POINTS[_i + 1]; _k++){

                        int _j = NLIST[_k];

                        int _cpp_halo_flag;
                        int _cp_halo_flag = 0;

                        if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                         %(KERNEL_ARGUMENT_DECL)s

                         //KERNEL CODE START

                         %(KERNEL)s

                         //KERNEL CODE END


                    }
                }
            }




            for(int _i = _DIV*_PBLOCK; _i < N_LOCAL; _i++){

                for(int _k = START_POINTS[_i]; _k < START_POINTS[_i+1]; _k++){
                    int _j = NLIST[_k];
                    int _cpp_halo_flag;
                    int _cp_halo_flag = 0;

                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }
            }

            %(LOOP_TIMER_POST)s

            return;
        }


        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        cell.cell_list.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict



        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Rebuild neighbour list potentially'''
        self._invocations += 1
        if cell.cell_list.version_id > self.neighbour_list.version_id:
            self.neighbour_list.update()
            self._neighbourlist_count += 1

        '''Create arg list'''
        _N_TOTAL = ctypes.c_int(self.neighbour_list.n_total)
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_TOTAL,
                 _N_LOCAL,
                 _STARTS,
                 _LIST]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()












# Unnecsary workaround
def _halo_exchange_particle_dat(dats_in):
    # loop through passed dats
    for ix in dats_in.values():

        # dats with existing access descriptors
        if type(ix) is tuple:

            # check is particle dat
            if type(ix[0]) is data.ParticleDat:

                # halo exchange if required
                if (len(ix) == 2) and (ix[1].read is True):
                    ix[0].halo_exchange()
                elif (len(ix) > 2) and (ix[1].read is True) and (ix[2] is True):
                    ix[0].halo_exchange()

        elif type(ix) is data.ParticleDat:
            ix.halo_exchange()



###############################################################################
# Neighbour list looping using NIII and pairwise lists
###############################################################################

class PairLoopNeighbourListPairIndices(_Base):
    def __init__(self, potential=None, dat_dict=None, kernel=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        self.loop_timer = opt.LoopTimer()

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)


        self.neighbour_list = cell.NeighbourListPairIndices()
        self.neighbour_list.setup(*cell.cell_list.get_setup_parameters())



        self._neighbourlist_count = 0
        self._invocations = 0

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        extern "C" void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int N_PAIRS, const int* __restrict__ LIST_I, const int* __restrict__ LIST_J, %(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _kernel_argument_declarations(self):
        s = '\n'
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return s

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include <stdio.h>
        
        #define _BLOCK_SIZE 8

        void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int N_PAIRS, const int* __restrict__ LIST_I, const int* __restrict__ LIST_J, %(ARGUMENTS)s) {
            
            const int _NBLOCKS = N_PAIRS / _BLOCK_SIZE;

            %(LOOP_TIMER_PRE)s
            

            for(int _px = 0; _px < _NBLOCKS*_BLOCK_SIZE; _px++){
                const int _i = LIST_I[_px];
                const int _j = LIST_J[_px];
            
                    int _cpp_halo_flag;
                    int _cp_halo_flag = 0;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

            }


            for(int _px = _NBLOCKS*_BLOCK_SIZE; _px < N_PAIRS; _px++){
                const int _i = LIST_I[_px];
                const int _j = LIST_J[_px];
            
                    int _cpp_halo_flag;
                    int _cp_halo_flag = 0;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

            }


            %(LOOP_TIMER_POST)s

            return;
        }


        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        cell.cell_list.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict



        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Rebuild neighbour list potentially'''
        self._invocations += 1
        if cell.cell_list.version_id > self.neighbour_list.version_id:
            self.neighbour_list.update()
            self._neighbourlist_count += 1

        '''Create arg list'''
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _N_PAIRS = self.neighbour_list.list_length
        _LIST_I = self.neighbour_list.listi.ctypes_data
        _LIST_J = self.neighbour_list.listj.ctypes_data

        args2 = [_N_LOCAL,
                 _N_PAIRS,
                 _LIST_I,
                 _LIST_J]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


