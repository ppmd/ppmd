import numpy as np
import particle
import ctypes
import os
import data
import loop
import build
import runtime
import access
import cell


class _Base(build.GenericToolChain):
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
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == particle.Dat:
                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == particle.TypedDat:

                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

################################################################################################################
# RAPAPORT LOOP SERIAL
################################################################################################################


class PairLoopRapaport(_Base):
    """
    Class to implement rapaport 14 cell looping.
    
    :arg int n: Number of elements to loop over.
    :arg domain domain: Domain containing the particles.
    :arg dat positions: Postitions of particles.
    :arg potential potential: Potential between particles.
    :arg dict dat_dict: Dictonary mapping between state vars and kernel vars.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, domain, potential, dat_dict):
        self._domain = domain
        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._compiler_set()

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)
        self._kernel = self._potential.kernel

        self._nargs = len(self._particle_dat_dict)

        self._code_init()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if runtime.MPI_HANDLE is None:
                self._create_library()
            else:
                if runtime.MPI_HANDLE.rank == 0:
                    self._create_library()
                runtime.MPI_HANDLE.barrier()

        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except OSError as e:
            raise OSError(e)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

    def _compiler_set(self):
        self._cc = build.TMPCC

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        
        
        void cell_index_offset(const unsigned int cp, const unsigned int cpp_i, int* cell_array, double *d_extent, unsigned int* cpp, unsigned int *flag, double *offset){
        
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
            //printf("cp=%%d, cpp_i=%%d, cpp=%%d, flag=%%d, offset[0]=%%f, offset[1]=%%f, offset[2]=%%f \\n ",
            //cp, cpp_i, *cpp, *flag, offset[0], offset[1], offset[2]);
            
                
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

                    /*
                    if (i==25) {
                        printf("CPU: i=%%d, cp=%%d ,cpp=%%d, s= %%f %%f %%f, p= %%f %%f %%f \\n", i, cp ,cpp, s[0], s[1], s[2], P_ext[25*3], P_ext[25*3+1], P_ext[25*3+2]);
                    }
                    */

                    while (i > -1){
                        j = q_list[n+cpp];
                        while (j > -1){
                            if (cp != cpp || i < j){
        
                                %(KERNEL_ARGUMENT_DECL)s
                                
                                  //KERNEL CODE START
                                  
                                  %(KERNEL)s
                                  
                                  //KERNEL CODE END
                                    /*
                                  if (i == 25 || j == 25) {
                                    printf("CPU r2=%%f \\n", r2);
                                  } */


                                
                                
                            }
                            j = q_list[j];  
                        }
                        i=q_list[i];
                    }
                }
            }
            
            //printf("CPU: i=%%d, A=%%f %%f %%f \\n", 25, A_ext[25*3], A_ext[25*3 + 1], A_ext[25*3 + 2]);
            return;
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
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == particle.Dat:
                if dat[1].name == 'positions':
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

                    s += space + 'if (flag){ \n'

                    # s += space+'double r1[3];\n'
                    s += space + 'r1[0] =' + argname + '[LINIDX_2D(3,j,0)] + s[0]; \n'
                    s += space + 'r1[1] =' + argname + '[LINIDX_2D(3,j,1)] + s[1]; \n'
                    s += space + 'r1[2] =' + argname + '[LINIDX_2D(3,j,2)] + s[2]; \n'
                    s += space + loc_argname + '[1] = r1;\n'

                    s += space + '}else{ \n'
                    s += space + loc_argname + '[1] = ' + argname + '+3*j;\n'
                    s += space + '} \n'
                    s += space + loc_argname + '[0] = ' + argname + '+3*i;\n'

                else:
                    ncomp = dat[1].ncomp
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                    s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == particle.TypedDat:

                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        
        void %(KERNEL_NAME)s_wrapper(const int n,const int cell_count, int* cells, int* q_list, double* d_extent,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        if n is not None:
            _N = n
        else:
            _N = cell.cell_list.cell_list[cell.cell_list.cell_list.end]

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Create arg list'''
        args = [ctypes.c_int(_N),
                ctypes.c_int(self._domain.cell_count),
                self._domain.cell_array.ctypes_data,
                cell.cell_list.cell_list.ctypes_data,
                self._domain.extent.ctypes_data]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat = dat_orig[0]
            else:
                dat = dat_orig
            args.append(dat.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)


################################################################################################################
# DOUBLE PARTICLE LOOP
################################################################################################################


class DoubleAllParticleLoop(loop.SingleAllParticleLoop):
    """
    Class to loop over all particle pairs once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg list headers: list containing C headers required by kernel.
    """

    def _compiler_set(self):
        self._cc = build.TMPCC

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

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
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == particle.Dat:
                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == particle.TypedDat:

                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

################################################################################################################
# DOUBLE PARTICLE LOOP APPLIES PBC
################################################################################################################


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
        self._nargs = len(self._particle_dat_dict)

        self._code_init()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if runtime.MPI_HANDLE is None:
                self._create_library()

            else:
                if runtime.MPI_HANDLE.rank == 0:
                    self._create_library()
                runtime.MPI_HANDLE.barrier()
        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

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
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n,double *extent_ext,%(ARGUMENTS)s);

        #endif
        '''

        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
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
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == particle.Dat:
                if dat[1].name == 'positions':
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

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
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                    s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == particle.TypedDat:

                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
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

        '''TODO IMPLEMENT/CHECK RESISTANCE TO ARG REORDERING'''

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat = dat_orig[0]
            else:
                dat = dat_orig
            args.append(dat.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']
        method(*args)


################################################################################################################
# RAPAPORT LOOP OPENMP
################################################################################################################


class PairLoopRapaportOpenMP(PairLoopRapaport):
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
        
        void cell_index_offset(const unsigned int cp, const unsigned int cpp_i, int* cell_array, double *d_extent, unsigned int* cpp, unsigned int *flag, double *offset){
        
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
        """Generate the source code the actual implementation.
        """

        d = {'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations_openmp(),
             'UNIQUENAME': self._unique_name,
             'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'OPENMP_INIT': self._ompinitstr,
             'OPENMP_DECLARATION': self._ompdecstr,
             'OPENMP_FINALISE': self._ompfinalstr
             }

        return self._code % d

    def _kernel_argument_declarations_openmp(self):
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
        space = ' ' * 14

        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW


            argname = dat[0] + '_ext'
            loc_argname = dat[0]

            reduction_handle = self._kernel.reduction_variable_lookup(dat[0])

            if reduction_handle is not None:
                assert dat[1].ncomp == 1, "Not valid for more than 1 element"

                # Create a var name a variable to reduce upon.
                reduction_argname = dat[0] + '_reduction'

                # Initialise variable
                self._ompinitstr += data.ctypes_map[dat[1].dtype] + ' ' + reduction_argname + ' = ' + \
                                    build.omp_operator_init_values[reduction_handle.operator] + ';'

                # Add to omp pragma
                self._ompdecstr += 'Reduction(' + reduction_handle.operator + ':' + reduction_argname + ')'

                # Modify kernel code to use new Reduction variable.
                self._kernel_code = build.replace(self._kernel_code, reduction_handle.pointer, reduction_argname)

                # write final value to output pointer

                self._ompfinalstr += argname + '[' + reduction_handle.index + '] =' + reduction_argname + ';'


            else:
                if type(dat[1]) == data.ScalarArray:
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

                elif type(dat[1]) == particle.Dat:
                    if dat[1].name == 'positions':
                        s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

                        s += space + 'if (flag){ \n'

                        # s += space+'double r1[3];\n'
                        s += space + 'r1[0] =' + argname + '[LINIDX_2D(3,j,0)] + s[0]; \n'
                        s += space + 'r1[1] =' + argname + '[LINIDX_2D(3,j,1)] + s[1]; \n'
                        s += space + 'r1[2] =' + argname + '[LINIDX_2D(3,j,2)] + s[2]; \n'
                        s += space + loc_argname + '[1] = r1;\n'

                        s += space + '}else{ \n'
                        s += space + loc_argname + '[1] = ' + argname + '+3*j;\n'
                        s += space + '} \n'
                        s += space + loc_argname + '[0] = ' + argname + '+3*i;\n'

                    else:
                        ncomp = dat[1].ncomp
                        s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                        s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                        s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

                elif type(dat[1]) == particle.TypedDat:

                    ncomp = dat[1].ncomp
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                    s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                        ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                    s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                        ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

################################################################################################################
# DOUBLE PARTICLE LOOP
################################################################################################################        

class DoubleAllParticleLoopOpenMP(DoubleAllParticleLoop):
    """
    Class to loop over all particle pairs once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg list headers: list containing C headers required by kernel.
    """

    def _compiler_set(self):
        self._cc = build.TMPCC

    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._ompinitstr = ''
        self._ompdecstr = ''
        self._ompfinalstr = ''

        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) {
          %(OPENMP_INIT)s
          
          #pragma omp parallel for schedule(dynamic) %(OPENMP_DECLARATION)s
          for (int i=0; i<n; i++) { 
            for (int j=0; j<i; j++) {  
              
              %(KERNEL_ARGUMENT_DECL)s
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
          }}
          %(OPENMP_FINALISE)s
        }
        '''

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations_openmp(),
             'UNIQUENAME': self._unique_name,
             'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'OPENMP_INIT': self._ompinitstr,
             'OPENMP_DECLARATION': self._ompdecstr,
             'OPENMP_FINALISE': self._ompfinalstr
             }
        return self._code % d

    def _kernel_argument_declarations_openmp(self):
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

            reduction_handle = self._kernel.reduction_variable_lookup(dat[0])

            if reduction_handle is not None:
                if dat[1].ncomp != 1:
                    print "WARNING, Reductions not valid for more than 1 element"

                # Create a var name a variable to reduce upon.
                reduction_argname = dat[0] + '_reduction'

                # Initialise variable
                self._ompinitstr += data.ctypes_map[dat[1].dtype] + ' ' + reduction_argname + ' = ' + \
                                    build.omp_operator_init_values[reduction_handle.operator] + ';'

                # Add to omp pragma
                self._ompdecstr += 'Reduction(' + reduction_handle.operator + ':' + reduction_argname + ')'

                # Modify kernel code to use new Reduction variable.
                self._kernel_code = build.replace(self._kernel_code, reduction_handle.pointer, reduction_argname)

                # write final value to output pointer

                self._ompfinalstr += argname + '[' + reduction_handle.index + '] =' + reduction_argname + ';'


            else:
                if type(dat[1]) == data.ScalarArray:
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

                elif type(dat[1]) == particle.Dat:
                    ncomp = dat[1].ncomp
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                    s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

                elif type(dat[1]) == particle.TypedDat:

                    ncomp = dat[1].ncomp
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                    s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                        ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                    s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                        ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

################################################################################################################
# RAPAPORT LOOP SERIAL PARTICLE LIST
################################################################################################################

class PairLoopRapaportParticleList(PairLoopRapaport):
    """
    Applies the Rapapport particle list method
    
    :arg int n: Number of elements to loop over.
    :arg domain domain: Domain containing the particles.
    :arg dat positions: Postitions of particles.
    :arg potential potential: Potential between particles.
    :arg dict dat_dict: Dictonary mapping between state vars and kernel vars.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, n, domain, positions, potential, dat_dict):

        self._N = n
        self._domain = domain
        self._potential = potential
        self._particle_dat_dict = dat_dict

        self._compiler_set()


        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)
        self._kernel = self._potential.kernel

        self._nargs = len(self._particle_dat_dict)

        self._code_init()
        self._cell_sort_setup()
        self._neighbour_list_setup()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if runtime.MPI_HANDLE is None:
                self._create_library()

            else:
                if runtime.MPI_HANDLE.rank == 0:
                    self._create_library()
                runtime.MPI_HANDLE.barrier()
        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

    def execute(self, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        self._cell_sort_all()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Create arg list'''
        args = [ctypes.c_int(self._N()),
                ctypes.c_int(self._domain.cell_count),
                self._domain.cell_array.ctypes_data,
                cell.cell_list.cell_list.ctypes_data,
                self._domain.extent.ctypes_data]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat = dat_orig[0]
            else:
                dat = dat_orig
            args.append(dat.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

################################################################################################################
# RAPAPORT LOOP SERIAL FOR HALO DOMAINS
################################################################################################################


class PairLoopRapaportHalo(PairLoopRapaport):
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

                if dat[1].name == 'potential_energy':

                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '; \n'
                    s += '\n'
                    s += space + 'if (cp_h_flag + cpp_h_flag >= 1){ \n'

                    # s+= space+'printf("cp = %d, cpp = %d ,cpf = %d, cppf = %d|", cp,cpp, cp_h_flag, cpp_h_flag);\n'

                    s += space + loc_argname + ' = &' + argname + '[1];\n'

                    s += space + '}else{ \n'
                    s += space + loc_argname + ' = ' + argname + ';\n'
                    s += space + '}\n'
                else:
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == particle.Dat:
                if dat[1].name == 'accelerations':
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

                    s += space + 'if (cp_h_flag > 0){ \n'
                    s += space + 'ri = null_array;\n'
                    s += space + '}else{ \n'
                    # if not in halo
                    s += space + 'ri = ' + argname + '+3*i;}\n'

                    s += '\n'

                    s += space + 'if (cpp_h_flag > 0){ \n'
                    s += space + 'rj = null_array;\n'
                    s += space + '}else{ \n'
                    # if not in halo
                    s += space + 'rj = ' + argname + '+3*j;}\n'

                    s += '\n'

                    s += space + loc_argname + '[1] = rj;\n'
                    s += space + loc_argname + '[0] = ri;\n'

                    s += '\n'



                else:
                    ncomp = dat[1].ncomp
                    s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    s += space + loc_argname + '[0] = ' + argname + '+' + str(ncomp) + '*i;\n'
                    s += space + loc_argname + '[1] = ' + argname + '+' + str(ncomp) + '*j;\n'

            elif type(dat[1]) == particle.TypedDat:

                ncomp = dat[1].ncomp
                s += space + data.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                s += space + loc_argname + '[0] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'
                s += space + loc_argname + '[1] = &' + argname + '[LINIDX_2D(' + str(
                    ncomp) + ',' + '_TYPE_MAP[j]' + ',0)];\n'

        return s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        
        void %(KERNEL_NAME)s_wrapper(const int n, int* cell_array, int* q_list,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
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
                    
                    
                    unsigned int cpp, cp_h_flag, cpp_h_flag; 
                    int i,j;
                    
                    
                    cell_index_offset(cp, cpp_i, cell_array, &cpp, &cp_h_flag, &cpp_h_flag);
                    
                    //Check that both cells are not halo cells.
                    
                    //printf("cpp=%%d, flagi=%%d, flagj=%%d |",cpp, cp_h_flag,cpp_h_flag);
                    
                    if ((cp_h_flag+cpp_h_flag) < 2){
                        
                        i = q_list[n+cp];
                        while (i > -1){
                            j = q_list[n+cpp];
                            while (j > -1){
                                if (cp != cpp || i < j){
                                    
                                    double *ri, *rj;
                                    
                                    double null_array[3] = {0,0,0};
                                    //printf("i=%%d, j=%%d |",i,j);
                                    
                                    
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
            }
            
            
            return;
        }        
        
        
        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Create arg list'''

        if n is not None:
            _N = n
        else:
            _N = cell.cell_list.cell_list[cell.cell_list.cell_list.end]

        args = [ctypes.c_int(_N),
                self._domain.cell_array.ctypes_data,
                cell.cell_list.cell_list.ctypes_data]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat = dat_orig[0]
            else:
                dat = dat_orig
            args.append(dat.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)
