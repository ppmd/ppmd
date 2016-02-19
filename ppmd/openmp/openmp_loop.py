
# system level
import numpy as np
import ctypes
import os

# package level
from ppmd import data
from ppmd import build
from ppmd import runtime
from ppmd import access
from ppmd import mpi
from ppmd import host
from ppmd import cell
from ppmd import loop



################################################################################################################
# SINGLE PARTICLE LOOP OPENMP
################################################################################################################


class SingleAllParticleLoopOpenMP(loop.SingleAllParticleLoop):
    """
    OpenMP version of single pass pair loop (experimental)
    """

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

        #include "%(LIB_DIR)s/generic.h"

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
             'OPENMP_FINALISE': self._ompfinalstr,
             'LIB_DIR': runtime.LIB_DIR.dir
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
                # if (dat[1].ncomp != 1):
                # print "WARNING, Reductions currently only valid for 1 element."

                # Create a var name a variable to reduce upon.
                reduction_argname = dat[0] + '_reduction'

                # Initialise variable
                self._ompinitstr += host.ctypes_map[dat[1].dtype] + ' ' \
                                    + reduction_argname \
                                    + ' = ' \
                                    + build.omp_operator_init_values[reduction_handle.operator] + ';'

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
                    s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';\n'
                    s += space + loc_argname + ' = ' + argname + '+' + str(ncomp) + '*i;\n'

                elif type(dat[1]) == data.TypedDat:

                    ncomp = dat[1].ncomp
                    s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';  \n'
                    s += space + loc_argname + ' = &' + argname + '[LINIDX_2D(' + str(
                        ncomp) + ',' + '_TYPE_MAP[i]' + ',0)];\n'

        return s
