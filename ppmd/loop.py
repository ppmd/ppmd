
# system level
import numpy as np
import ctypes
#import os

# package level
import data
import build
import runtime
import access
import host
import generation

class _Base(object):
    """
    Generic base class to loop over all particles once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg bool DEBUG: Flag to enable debug flags.
    """

    def __init__(self, n, types_map, kernel, particle_dat_dict):

        self._cc = build.TMPCC

        self._N = n
        self._types_map = types_map

        self._kernel = kernel

        self._particle_dat_dict = particle_dat_dict
        self._nargs = len(self._particle_dat_dict)

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
        # s = '\n'

        # Code block to hold the macros for the mapping from symbols to pointers
        _map = build.Code()

        for i, dat_orig in enumerate(self._particle_dat_dict.items()):
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW
            space = ' ' * 14
            _nl = '\n'

            # This is the symbol used for the function decleration
            argname = dat[0] + '_ext'

            # symbol used in the kernel
            loc_argname = dat[0]

            # undef any previous macros for this symbol
            _map += '#undef ' + loc_argname + _nl

            # If the symbol corresponds to an array we want a direct map
            if (type(dat[1]) == data.ScalarArray) or (type(dat[1]) == host.PointerArray):
                _map += '#define ' + loc_argname + '(x) ' + argname + '[(x)]' + _nl

            # if the symbol is a particle dat then the first index and only
            # index maps into the correct column.
            if type(dat[1]) == data.ParticleDat:
                ncomp = dat[1].ncomp
                _map += '#define ' + loc_argname + '(x) ' + argname + '[' + generation.get_first_index_symbol() + '*' + str(ncomp) + ' + (x)]' + _nl

            if type(dat[1]) == data.TypedDat:
                ncomp = dat[1].ncol
                _map += '#define ' + loc_argname + '(x) ' + argname + '[LINIDX_2D(' + str(
                     ncomp) + ',' + '(x), ' + '_TYPE_MAP[' + generation.get_first_index_symbol() + '])]' + _nl

        return _map

    def _argnames(self):
        """Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of
        the method which executes the pairloop over the grid.
        If, for example, the pairloop gets passed two particle_dats,
        then the result will be ``double** arg_000,double** arg_001`.`
        """

        #self._argtypes = []

        argnames = ''
        if self._kernel.static_args is not None:
            self._static_arg_order = []

            for i, dat in enumerate(self._kernel.static_args.items()):
                argnames += 'const ' + host.ctypes_map[dat[1]] + ' ' + dat[0] + ','
                self._static_arg_order.append(dat[0])


        for i, dat in enumerate(self._particle_dat_dict.items()):
            if type(dat[1]) is tuple:
                _dtype = dat[1][0].dtype
                _mode = dat[1][1]
            else:
                _dtype = dat[1].dtype
                _mode = access.RW

            if not _mode.write:
                _const = 'const '
            else:
                _const = ''

            argnames += _const + host.ctypes_map[_dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'


        return argnames[:-1]

    def _loc_argnames(self):
        """Comma separated string of local argument names.
        """
        argnames = ''
        for i, dat in enumerate(self._particle_dat_dict.items()):
            # dat[0] is always the name, even with access descriptiors.
            argnames += dat[0] + ','
        return argnames[:-1]

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        #define _RESTRICT %(RESTRICT)s

        extern "C" void %(KERNEL_NAME)s_wrapper(int n,%(ARGUMENTS)s);

        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir,
             'RESTRICT':self._cc.restrict_keyword}
        return code % d

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + x + '\" \n'
        return s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'INDEX_I': generation.get_first_index_symbol(),
             'KERNEL': self._kernel.code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations()}

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


################################################################################################################
# SINGLE ALL PARTICLE LOOP SERIAL
################################################################################################################


class ParticleLoop(_Base):

    def _code_init(self):

        self._code = '''

        void %(KERNEL_NAME)s_wrapper(const int _N, int* _RESTRICT _TYPE_MAP,%(ARGUMENTS)s) {

            for (int %(INDEX_I)s=0; %(INDEX_I)s<_N; %(INDEX_I)s++) {
                %(KERNEL_ARGUMENT_DECL)s

                    //KERNEL CODE START

                    %(KERNEL)s

                    //KERNEL CODE END
            }

        }
        '''

    # added to cope with int *_GID, int *_TYPE_MAP, take out afterwards
    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """

        code = '''
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        #define _RESTRICT %(RESTRICT)s

        extern "C" void %(KERNEL_NAME)s_wrapper(const int _N, int* _RESTRICT _TYPE_MAP,%(ARGUMENTS)s);
        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir,
             'RESTRICT':self._cc.restrict_keyword}
        return code % d


################################################################################################################
# SINGLE PARTICLE LOOP SERIAL
################################################################################################################


class LimitedParticleLoop(_Base):

    def _code_init(self):

        self._code = '''

        void %(KERNEL_NAME)s_wrapper(const int _START_IX, const int _END_IX, int* _RESTRICT _TYPE_MAP, %(ARGUMENTS)s) {
          for (int %(INDEX_I)s=_START_IX; %(INDEX_I)s<_END_IX; %(INDEX_I)s++) {
              %(KERNEL_ARGUMENT_DECL)s
              
                  //KERNEL CODE START
                  
                  %(KERNEL)s
                  
                  //KERNEL CODE END
              
            }
        }
        '''

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        #define _RESTRICT %(RESTRICT)s

        extern "C" void %(KERNEL_NAME)s_wrapper(const int _START_IX, const int _END_IX, int* _RESTRICT _TYPE_MAP, %(ARGUMENTS)s);

        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir,
             'RESTRICT':self._cc.restrict_keyword}
        return code % d

    def execute(self, start=None, end=None, dat_dict=None, static_args=None):

        # Allow alternative pointers
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict


        args = [ctypes.c_int(start), ctypes.c_int(end)]


        if self._types_map is not None:
            args.append(self._types_map.ctypes_data)
        else:
            _null_type_map = ctypes.c_int()
            args.append(ctypes.byref(_null_type_map))


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


