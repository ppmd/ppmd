from __future__ import print_function, division, absolute_import

from ppmd import opt, runtime, host
from ppmd.lib.build import simple_lib_creator, TMPCC, TMPCC_OpenMP

__author__      = "W.R.Saunders"
__copyright__   = "Copyright 2016, W.R.Saunders"


class SharedLib(object):
    """
    Generic base class to loop over all particles once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict dat_dict: Dictonary storing map between kernel variables
    and state variables.
    :arg bool runtime.DEBUG: Flag to enable runtime.DEBUG flags.
    """

    def __init__(self, kernel, dat_dict, openmp=False):

        # Timers
        self.creation_timer = opt.Timer(runtime.BUILD_TIMER, 2, start=True)
        """Timer that times the creation of the shared library if
        runtime.BUILD_TIMER > 2"""

        self.execute_timer = opt.Timer(runtime.TIMER, 0, start=False)
        """Timer that times the execution time of the shared library if
        runtime.BUILD_TIMER > 2"""

        self.execute_overhead_timer = opt.Timer(runtime.BUILD_TIMER, 2,
                                                    start=False)
        """Times the overhead required before the shared library is ran if
        runtime.BUILD_TIMER > 2. """

        self._omp = openmp

        self._compiler_set()
        self._temp_dir = runtime.BUILD_DIR

        self._kernel = kernel

        self._dat_dict = dat_dict
        self._nargs = len(self._dat_dict)

        self._code_init()

        self._lib = simple_lib_creator(self._generate_header_source(),
                                       self._generate_impl_source(),
                                       self._kernel.name,
                                       CC=self._cc)


        self.creation_timer.stop("SharedLib creation timer " +
                                 str(self._kernel.name))

    def _compiler_set(self):
        if self._omp is False:
            self._cc = TMPCC
        else:
            self._cc = TMPCC_OpenMP

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

        for i, dat in enumerate(self._dat_dict.items()):

            space = ' ' * 14
            argname = dat[0]  # +'_ext'
            loc_argname = argname  # dat[0]

            if issubclass(type(dat[1]), host.Array):
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + \
                     loc_argname + ' = ' + argname + ';\n'

            if issubclass(type(dat[1]), host.Matrix):
                ncomp = dat[1].ncol
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + \
                     loc_argname + ';\n'
                s += space + loc_argname + ' = ' + argname + '+' + \
                     str(ncomp) + '*i;\n'

        return s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        extern "C" void %(KERNEL_NAME)s_wrapper(%(ARGUMENTS)s);

        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR}
        return code % d

    def execute(self, dat_dict=None, static_args=None):
        # Timing block 1
        self.execute_overhead_timer.start()


        """Allow alternative pointers"""
        if dat_dict is not None:

            for key in self._dat_dict:
                self._dat_dict[key] = dat_dict[key]

        args = []

        '''TODO IMPLEMENT/CHECK RESISTANCE TO ARG REORDERING'''

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            for dat in static_args.values():
                args.append(dat)


        '''Add pointer arguments to launch command'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)


        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        # Timing block 2
        self.execute_overhead_timer.pause()
        self.execute_timer.start()


        return_code = method(*args)

        # Timing block 3
        self.execute_timer.pause()
        self.execute_overhead_timer.start()

        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()
        # Timing block 4
        self.execute_overhead_timer.pause()


        return return_code

    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''

        void %(KERNEL_NAME)s_wrapper(%(ARGUMENTS)s) {
          
          %(KERNEL)s
        
        }
        '''

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
                argnames += '' + host.ctypes_map[dat[1]] + ' ' + dat[0] + ','
                self._static_arg_order.append(dat[0])
                #self._argtypes.append(dat[1])


        for i, dat in enumerate(self._dat_dict.items()):
            if type(dat[1]) is not tuple:
                argnames += host.ctypes_map[dat[1].dtype] + ' * ' + \
                            self._cc.restrict_keyword + ' ' + dat[0] + ','
            else:
                argnames += host.ctypes_map[dat[1][0].dtype] + ' * ' + \
                            self._cc.restrict_keyword + ' ' + dat[0] + ','

        return argnames[:-1]

    def _loc_argnames(self):
        """Comma separated string of local argument names.
        """
        argnames = ''
        for i, dat in enumerate(self._dat_dict.items()):
            # dat[0] is always the name, even with access descriptiors.
            argnames += dat[0] + ','
        return argnames[:-1]

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + str(x) + '\" \n'
        return s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations()}

        return self._code % d




