import numpy as np
import ctypes
import os
import cpu_generate_generic
import data
import build
import runtime
import access
import cell
import mpi
import host
import cpu_generate_openmp


class _Base(build.GenericToolChain):

    def __init__(self, n, types_map, kernel, particle_dat_dict):

        self._compiler_set()
        self._N = n
        self._types_map = types_map

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

            if mpi.MPI_HANDLE is None:
                self._create_library()

            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()

        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

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

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n, int *_TYPE_MAP,%(ARGUMENTS)s);

        #endif
        '''

        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

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

    def __init__(self, domain, potential=None, dat_dict=None, kernel=None):
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

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."


        self._nargs = len(self._particle_dat_dict)

        self._code_init()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if mpi.MPI_HANDLE is None:
                self._create_library()
            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()

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
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            elif type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'positions':
                    s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

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
        cell.cell_list.check()
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

################################################################################################################
# DOUBLE PARTICLE LOOP
################################################################################################################


class DoubleAllParticleLoop(_Base):
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
            if mpi.MPI_HANDLE is None:
                self._create_library()

            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()
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
                self._ompinitstr += host.ctypes_map[dat[1].dtype] + ' ' + reduction_argname + ' = ' + \
                                    build.omp_operator_init_values[reduction_handle.operator] + ';'

                # Add to omp pragma
                self._ompdecstr += 'Reduction(' + reduction_handle.operator + ':' + reduction_argname + ')'

                # Modify kernel code to use new Reduction variable.
                self._kernel_code = build.replace(self._kernel_code, reduction_handle.pointer, reduction_argname)

                # write final value to output pointer

                self._ompfinalstr += argname + '[' + reduction_handle.index + '] =' + reduction_argname + ';'


            else:
                if type(dat[1]) == data.ScalarArray:
                    s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

                elif type(dat[1]) == data.ParticleDat:
                    if dat[1].name == 'positions':
                        s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

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
                self._ompinitstr += host.ctypes_map[dat[1].dtype] + ' ' + reduction_argname + ' = ' + \
                                    build.omp_operator_init_values[reduction_handle.operator] + ';'

                # Add to omp pragma
                self._ompdecstr += 'Reduction(' + reduction_handle.operator + ':' + reduction_argname + ')'

                # Modify kernel code to use new Reduction variable.
                self._kernel_code = build.replace(self._kernel_code, reduction_handle.pointer, reduction_argname)

                # write final value to output pointer

                self._ompfinalstr += argname + '[' + reduction_handle.index + '] =' + reduction_argname + ';'


            else:
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



################################################################################################################
# RAPAPORT LOOP SERIAL FOR HALO DOMAINS
################################################################################################################


class PairLoopRapaportHalo(PairLoopRapaport):


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


            s += cpu_generate_generic.generate_map(pair=True,
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

        '''Halo exchange'''
        _halo_exchange_particle_dat(self._particle_dat_dict)

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

################################################################################################################
# PairLoopRapaportHaloOpenMP LOOP SERIAL FOR HALO DOMAINS
################################################################################################################


class PairLoopRapaportHaloOpenMP(PairLoopRapaport):
    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP

    def cell_offset_mapping(self):
        """
        Calculate cell offset mappings

        :return:
        """

        _map_array = host.Array(ncomp=27, dtype=ctypes.c_int)

        _map = ((-1,1,-1),
                (-1,-1,-1),
                (-1,0,-1),
                (0,1,-1),
                (0,-1,-1),
                (0,0,-1),
                (1,0,-1),
                (1,1,-1),
                (1,-1,-1),

                (-1,1,0),
                (-1,0,0),
                (-1,-1,0),
                (0,-1,0),
                (0,0,0),
                (0,1,0),
                (1,0,0),
                (1,1,0),
                (1,-1,0),

                (-1,0,1),
                (-1,1,1),
                (-1,-1,1),
                (0,0,1),
                (0,1,1),
                (0,-1,1),
                (1,0,1),
                (1,1,1),
                (1,-1,1))


        for ix in range(27):

            _map_array[ix] = _map[ix][0] + _map[ix][1] * self._domain.cell_array[0] + _map[ix][2] * self._domain.cell_array[0]* self._domain.cell_array[1]

        return _map_array

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include <stdio.h>
        #include "%(LIB_DIR)s/generic.h"
        #include <omp.h>

        void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int* cell_map, const int n, int* cell_array, int* q_list,%(ARGUMENTS)s);

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

        void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int* cell_map, const int n, int* cell_array, int* q_list,%(ARGUMENTS)s) {


            //omp_set_num_threads(4);
            #pragma omp parallel
            { // parallel start

                %(LOOPING_ARGUMENT_DECL)s


                #pragma omp for
                for(int cax = 1; cax < cell_array[0]-1; cax++){
                for(int cay = 1; cay < cell_array[1]-1; cay++){
                for(int caz = 1; caz < cell_array[2]-1; caz++){

                    //printf("%%d %%d \\n", omp_get_thread_num(), omp_get_num_threads());

                    int cp  = (caz*cell_array[1] + cay)*cell_array[0] + cax;

                    for(int cpp_i=0; cpp_i<27; cpp_i++){
                        int cpp = cp + cell_map[cpp_i];

                        int _i,_j;

                        _i = q_list[n+cp];
                        while (_i > -1){
                            _j = q_list[n+cpp];
                            while (_j > -1){
                                if (_i != _j){
                                    int _cpp_halo_flag;
                                    int _cp_halo_flag;

                                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                                    if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

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


                }}} //triple loop end.


            %(OPENMP_LOOPING_FINALISE)s




            } //parallel end

            return;
        }
        '''

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """
        _kernel_argument_declarations = '\n'
        _looping_argument_declarations = '\n'
        _looping_argument_finalise = '\n'

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

            _kernel_argument_declarations += cpu_generate_openmp.generate_map(pair=True,
                                                                              symbol_external=dat[0] + '_ext',
                                                                              symbol_internal=dat[0],
                                                                              dat=dat[1],
                                                                              access_type=_mode,
                                                                              n3_disable_dats=_dd)
            _looping_argument_declarations += cpu_generate_openmp.generate_reduction_init_stage(symbol_external=dat[0] + '_ext',
                                                                                                symbol_internal=dat[0],
                                                                                                dat=dat[1],
                                                                                                access_type=_mode)
            _looping_argument_finalise += cpu_generate_openmp.generate_reduction_final_stage(symbol_external=dat[0] + '_ext',
                                                                                             symbol_internal=dat[0],
                                                                                             dat=dat[1],
                                                                                             access_type=_mode)

        d = {'UNIQUENAME': self._unique_name,
             'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': _kernel_argument_declarations,
             'LOOPING_ARGUMENT_DECL': _looping_argument_declarations,
             'OPENMP_LOOPING_FINALISE': _looping_argument_finalise}

        return self._code % d

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """
        cell.cell_list.check()
        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict


        '''Create arg list'''
        if n is not None:
            _N = n
        else:
            _N = cell.cell_list.cell_list[cell.cell_list.cell_list.end]

        _map_array = self.cell_offset_mapping()


        args = [ctypes.c_int(cell.cell_list.num_particles),
                _map_array.ctypes_data,
                ctypes.c_int(_N),
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






################################################################################################################
# Neighbour list looping using NIII
################################################################################################################

class PairLoopNeighbourList(_Base):
    def __init__(self, potential=None, dat_dict=None, kernel=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._compiler_set()

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


        self._nargs = len(self._particle_dat_dict)

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if mpi.MPI_HANDLE is None:
                self._create_library()
            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()

        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except OSError as e:
            raise OSError(e)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

        self.neighbour_list = cell.NeighbourList()
        self.neighbour_list.setup(*cell.cell_list.get_setup_parameters())


    def _compiler_set(self):
        self._cc = build.TMPCC

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* START_POINTS, const int* NLIST, %(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
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


            s += cpu_generate_generic.generate_map(pair=True,
                                                   symbol_external=dat[0] + '_ext',
                                                   symbol_internal=dat[0],
                                                   dat=dat[1],
                                                   access_type=_mode)

        return s

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <stdio.h>

        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* START_POINTS, const int* NLIST, %(ARGUMENTS)s) {

            for(int _i = 0; _i < N_LOCAL; _i++){
                for(int _k = START_POINTS[_i]; _k < START_POINTS[_i+1]; _k++){
                    int _j = NLIST[_k];
                    int _cpp_halo_flag;
                    int _cp_halo_flag;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

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
        if cell.cell_list.version_id > self.neighbour_list.version_id:
            self.neighbour_list.update()

        '''Create arg list'''
        _N_TOTAL = ctypes.c_int(self.neighbour_list.n_total)
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_TOTAL,
                 _N_LOCAL,
                 _STARTS,
                 _LIST]

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





################################################################################################################
# Neighbour list looping using NIII
################################################################################################################

class PairLoopNeighbourListOpenMP(PairLoopNeighbourList):
    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP

    def __init__(self, potential=None, dat_dict=None, kernel=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._compiler_set()

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


        self._nargs = len(self._particle_dat_dict)

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if mpi.MPI_HANDLE is None:
                self._create_library()
            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()

        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except OSError as e:
            raise OSError(e)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

        # Create an instance of a non N3 neighbour list.
        self.neighbour_list_non_n3 = cell.NeighbourListNonN3()
        self.neighbour_list_non_n3.setup(*cell.cell_list.get_setup_parameters())

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        #include <omp.h>

        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* START_POINTS, const int* NLIST, %(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
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


            s += cpu_generate_generic.generate_map(pair=True,
                                                   symbol_external=dat[0] + '_ext',
                                                   symbol_internal=dat[0],
                                                   dat=dat[1],
                                                   access_type=_mode)

        return s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """
        _kernel_argument_declarations = '\n'
        _looping_argument_declarations = '\n'
        _looping_argument_finalise = '\n'

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

            _kernel_argument_declarations += cpu_generate_openmp.generate_map(pair=True,
                                                                              symbol_external=dat[0] + '_ext',
                                                                              symbol_internal=dat[0],
                                                                              dat=dat[1],
                                                                              access_type=_mode,
                                                                              n3_disable_dats=_dd)
            _looping_argument_declarations += cpu_generate_openmp.generate_reduction_init_stage(symbol_external=dat[0] + '_ext',
                                                                                                symbol_internal=dat[0],
                                                                                                dat=dat[1],
                                                                                                access_type=_mode)
            _looping_argument_finalise += cpu_generate_openmp.generate_reduction_final_stage(symbol_external=dat[0] + '_ext',
                                                                                             symbol_internal=dat[0],
                                                                                             dat=dat[1],
                                                                                             access_type=_mode)

        d = {'UNIQUENAME': self._unique_name,
             'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': _kernel_argument_declarations,
             'LOOPING_ARGUMENT_DECL': _looping_argument_declarations,
             'OPENMP_LOOPING_FINALISE': _looping_argument_finalise}

        return self._code % d

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <stdio.h>

        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* START_POINTS, const int* NLIST, %(ARGUMENTS)s) {

            #pragma omp parallel
            {

                %(LOOPING_ARGUMENT_DECL)s

                #pragma omp for schedule(static, 8)
                for(int _i = 0; _i < N_LOCAL; _i++){
                    for(int _k = START_POINTS[_i]; _k < START_POINTS[_i+1]; _k++){
                        int _j = NLIST[_k];
                        int _cpp_halo_flag;
                        int _cp_halo_flag;

                        // set halo flag, TODO move all halo flags to be an if condition on particle index?
                        if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                        if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                         %(KERNEL_ARGUMENT_DECL)s

                             //KERNEL CODE START

                             %(KERNEL)s

                             //KERNEL CODE END

                    }
                }

            %(OPENMP_LOOPING_FINALISE)s

            } // parallel end

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
        if cell.cell_list.version_id > self.neighbour_list_non_n3.version_id:
            # print "REBUILDING"
            self.neighbour_list_non_n3.update()
        else:
            pass

        '''Create arg list'''
        _N_TOTAL = ctypes.c_int(self.neighbour_list_non_n3.n_total)
        _N_LOCAL = ctypes.c_int(self.neighbour_list_non_n3.n_local)
        _STARTS = self.neighbour_list_non_n3.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list_non_n3.list.ctypes_data

        args2 = [_N_TOTAL,
                 _N_LOCAL,
                 _STARTS,
                 _LIST]

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






#############################################################################################
# Layer based 1
#############################################################################################

class PairLoopNeighbourListLayersHybrid(_Base):

    def _compiler_set(self):
        self._cc = build.TMPCC_OpenMP

    def __init__(self, potential=None, dat_dict=None, kernel=None, openmp=False):

        self._potential = potential
        self._particle_dat_dict = dat_dict

        self._omp = openmp
        self._compiler_set()

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."


        self._nargs = len(self._particle_dat_dict)

        # Init code
        self._kernel_code = self._kernel.code

        
        self._code_init()

        self._unique_name = self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if mpi.MPI_HANDLE is None:
                self._create_library()
            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()

        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except OSError as e:
            raise OSError(e)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

        

        self.layer_method = cell.CellLayerSort()
        self.layer_method.setup(cell.cell_list, self._omp)

        self.neighbour_method = cell.NeighbourListLayerBased()
        self.neighbour_method.setup(self.layer_method, cell.cell_list, self._omp)

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        #include <omp.h>

        void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int Nn, const int* NMATRIX, %(ARGUMENTS)s);

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

        void %(KERNEL_NAME)s_wrapper(const int N_LOCAL, const int Nn, const int* NMATRIX, %(ARGUMENTS)s) {

            #pragma omp parallel
            {
            %(LOOPING_ARGUMENT_DECL)s
            
                #pragma omp for schedule(dynamic)
                for(int _i = 0; _i < N_LOCAL; _i++){
                for(int _k = _i*(Nn+1)+1; _k < (_i*Nn+1)+1 + NMATRIX[_i*(Nn+1)]; _k++){

                    int _j = NMATRIX[_k];
                    int _cpp_halo_flag;
                    int _cp_halo_flag;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }
            }
           

            %(OPENMP_LOOPING_FINALISE)s


            } //omp parallel end
            return;
        }


        '''



    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """
        _kernel_argument_declarations = '\n'
        _looping_argument_declarations = '\n'
        _looping_argument_finalise = '\n'

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

            _kernel_argument_declarations += cpu_generate_openmp.generate_map(pair=True,
                                                                              symbol_external=dat[0] + '_ext',
                                                                              symbol_internal=dat[0],
                                                                              dat=dat[1],
                                                                              access_type=_mode,
                                                                              n3_disable_dats=_dd)
            _looping_argument_declarations += cpu_generate_openmp.generate_reduction_init_stage(symbol_external=dat[0] + '_ext',
                                                                                                symbol_internal=dat[0],
                                                                                                dat=dat[1],
                                                                                                access_type=_mode)
            _looping_argument_finalise += cpu_generate_openmp.generate_reduction_final_stage(symbol_external=dat[0] + '_ext',
                                                                                             symbol_internal=dat[0],
                                                                                             dat=dat[1],
                                                                                             access_type=_mode)

        d = {'UNIQUENAME': self._unique_name,
             'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': _kernel_argument_declarations,
             'LOOPING_ARGUMENT_DECL': _looping_argument_declarations,
             'OPENMP_LOOPING_FINALISE': _looping_argument_finalise}

        return self._code % d

