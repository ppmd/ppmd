import numpy as np
import host
import ctypes
import os
import hashlib
import subprocess
import re
import runtime
import mpi


################################################################################################################
# COMPILERS START
################################################################################################################


class Compiler(object):
    """
    Container to define different compilers.
    
    :arg str name: Compiler name, referance only.
    :arg str binary: Name(+path) of Compiler binary.
    :arg list c_flags: List of compile flags as strings.
    :arg list l_flags: List of link flags as strings.
    :arg list opt_flags: List of optimisation flags.
    :arg list dbg_flags: List of runtime.DEBUG flags as strings.
    :arg list compile_flag: List of compile flag as single string (eg ['-c'] for gcc).
    :arg list shared_lib_flag: List of flags as strings to link as shared library.
    :arg string restrict_keyword: keyword to use for non aliased pointers
    """

    def __init__(self, name, binary, c_flags, l_flags, opt_flags, dbg_flags, compile_flag, shared_lib_flag, restrict_keyword=''):
        self._name = name
        self._binary = binary
        self._cflags = c_flags
        self._lflags = l_flags
        self._optflags = opt_flags
        self._dbgflags = dbg_flags
        self._compileflag = compile_flag
        self._sharedlibf = shared_lib_flag
        self._restrictkeyword = restrict_keyword

    @property
    def restrict_keyword(self):
        return self._restrictkeyword

    @property
    def name(self):
        """Return Compiler name."""
        return self._name

    @property
    def binary(self):
        """Return Compiler binary."""
        return self._binary

    @property
    def c_flags(self):
        """Return Compiler compile flags"""
        return self._cflags

    @property
    def l_flags(self):
        """Return Compiler link flags"""
        return self._lflags

    @property
    def opt_flags(self):
        """Return Compiler runtime.DEBUG flags"""
        return self._optflags

    @property
    def dbg_flags(self):
        """Return Compiler runtime.DEBUG flags"""
        return self._dbgflags

    @property
    def compile_flag(self):
        """Return Compiler compile flag."""
        return self._compileflag

    @property
    def shared_lib_flag(self):
        """Return Compiler link as shared library flag."""
        return self._sharedlibf

        # Define system gcc version as Compiler.


GCC = Compiler(['GCC'],
               ['gcc'],
               ['-fPIC', '-std=c99'],
               ['-lm'],
               ['-O3', '-march=native', '-m64'],
               ['-g'],
               ['-c'],
               ['-shared'],
               '__restrict__')

# Define system gcc version as OpenMP Compiler.
GCC_OpenMP = Compiler(['GCC'],
                      ['gcc'],
                      ['-fopenmp', '-fPIC', '-std=c99'],
                      ['-lgomp', '-lrt', '-Wall'],
                      ['-O3', '-march=native', '-m64'],
                      ['-g'],
                      ['-c', '-Wall'],
                      ['-shared'],
                      '__restrict__')


# Define system icc version as Compiler.
ICC = Compiler(['ICC'],
               ['icc'],
               ['-fpic', '-std=c99'],
               ['-lm'],
               ['-O3', '-xHost', '-restrict', '-m64'],
               ['-g'],
               ['-c'],
               ['-shared'],
               'restrict')

# Define system icc version as OpenMP Compiler.
ICC_OpenMP = Compiler(['ICC'],
                      ['icc'],
                      ['-fpic', '-openmp', '-std=c99'],
                      ['-openmp', '-lgomp', '-lpthread', '-lc', '-lrt'],
                      ['-O3', '-xHost', '-restrict', '-m64'],
                      ['-g', '-xHost', '-restrict', '-m64'],
                      ['-c'],
                      ['-shared'],
                      'restrict')


# Temporary Compiler flag
ICC_LIST = ['mapc-4044']

if os.uname()[1] in ICC_LIST:
    TMPCC = ICC
    TMPCC_OpenMP = ICC_OpenMP
    # TMPCC = GCC
    # TMPCC_OpenMP = GCC_OpenMP
else:
    TMPCC = GCC
    TMPCC_OpenMP = GCC_OpenMP


################################################################################################################
# OPENMP TOOLS START
################################################################################################################

def replace_dict(code, new_dict):
    for x in new_dict.items():
        regex = '(?<=[\W])(' + x[0] + ')(?=[\W])'
        code = re.sub(regex, str(x[1]), code)
    return code


def replace(code, old, new):
    old = old.replace('[', '\[')
    old = old.replace(']', '\]')

    regex = '(?<=[\W])(' + old + ')(?=[\W])'

    code = re.sub(regex, str(new), code)

    return code


# OpenMP Reduction definitions
omp_operator_init_values = {'+': '0', '-': '0', '*': '1', '&': '~0', '|': '0', '^': '0', '&&': '1', '||': '0'}


################################################################################################################
# AUTOCODE TOOLS START
################################################################################################################


def load_library_exception(kernel_name='None supplied', unique_name='None supplied', looping_type='None supplied'):
    """
    Attempts to create useful error messages for code generation.
    
    :arg str kernel_name: Name of kernel
    :arg str unique_name: Unique name given to kernel.
    :arg loop looping_type: Loop/Pairloop applied to kernel.
    """
    err_msg = "Could not read error file."
    err_read = False
    err_line = -1
    err_code = "Source not read."

    # Try to open error file.
    try:
        f = open(runtime.BUILD_DIR.dir + unique_name + '.err', 'r')
        err_msg = f.read()
        f.close()
        err_read = True

    except:
        print "Error file not read"

    # Try to read source lines around error.
    if err_read:
        m = re.search('[0-9]+:[0-9]', err_msg)
        try:
            m = re.search('[0-9]+:', m.group(0))
        except:
            pass
        try:
            err_line = int(m.group(0)[:-1])
        except:
            pass
        if err_line > 0:
            try:
                f = open(runtime.BUILD_DIR.dir + unique_name + '.c', 'r')
                code_str = f.read()
                f.close()
            except:
                print "Source file not read"

            code_str = code_str.split('\n')[max(0, err_line - 6):err_line + 1]
            code_str[-3] += "    <-------------"
            code_str = [x + "\n" for x in code_str]

            err_code = ''.join(code_str)
    print "Unique name", unique_name, "Rank", mpi.MPI_HANDLE.rank

    raise RuntimeError("\n"
                       "###################################################### \n"
                       "\t \t \t ERROR \n"
                       "###################################################### \n"
                       "kernel name: " + str(kernel_name) + "\n"
                       "------------------------------------------------------ \n"
                       "looping class: " + str(looping_type) + "\n"
                       "------------------------------------------------------ \n"
                       "Compile/link error message: \n \n" +
                       str(err_msg) + "\n"
                       "------------------------------------------------------ \n"
                       "Error location attempt: \n \n" +
                       str(err_code) + "\n \n"
                       "###################################################### \n"
                       )


def loop_unroll(str_start, i, j, step, str_end=None, key=None):
    """
    Function to create unrolled loops in python source. Potentialy add auto padding here?
    
    :arg str str_start: Starting string.
    :arg int i: Start index.
    :arg int j: End index.
    :arg int step: Stepsize between i and j.
    :arg str str_end: Ending string.
    """
    _s = ''

    if key is None:
        for ix in range(i, j, step):
            _s += str(str_start) + str(ix) + str(str_end)
    else:
        _regex = '(?<=[\W])(' + key + ')(?=[\W])'
        for ix in range(i, j + 1, step):
            _s += re.sub(_regex, str(ix), str_start) + '\n'

    return _s


################################################################################################################
# GENERIC TOOL CHAIN LOOPING
################################################################################################################


class GenericToolChain(object):
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
            if type(dat[1]) is not tuple:
                argnames += host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'
            else:
                argnames += host.ctypes_map[dat[1][0].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'


        return argnames[:-1]

    def _loc_argnames(self):
        """Comma separated string of local argument names.
        """
        argnames = ''
        for i, dat in enumerate(self._particle_dat_dict.items()):
            # dat[0] is always the name, even with access descriptiors.
            argnames += dat[0] + ','
        return argnames[:-1]

    def _unique_name_calc(self):
        """Return name which can be used to identify the pair loop
        in a unique way.
        """
        return self._kernel.name + '_' + self.hexdigest()

    def hexdigest(self):
        """Create unique hex digest"""
        m = hashlib.md5()
        m.update(self._kernel.code + self._code)
        if self._kernel.headers is not None:
            for header in self._kernel.headers:
                m.update(header)
        return m.hexdigest()

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/generic.h"
        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n,%(ARGUMENTS)s);

        #endif
        '''

        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + x + '\" \n'
        return s

    def _create_library(self):
        """
        Create a shared library from the source code.
        """

        filename_base = os.path.join(self._temp_dir, self._unique_name)

        header_filename = filename_base + '.h'
        impl_filename = filename_base + '.c'
        with open(header_filename, 'w') as f:
            print >> f, self._generate_header_source()
        with open(impl_filename, 'w') as f:
            print >> f, self._generate_impl_source()

        object_filename = filename_base + '.o'
        library_filename = filename_base + '.so'

        if runtime.VERBOSE.level > 2:
            print "Building", library_filename

        cflags = []
        cflags += self._cc.c_flags

        if runtime.DEBUG.level > 0:
            cflags += self._cc.dbg_flags
        else:
            cflags += self._cc.opt_flags


        cc = self._cc.binary
        ld = self._cc.binary
        lflags = self._cc.l_flags

        compile_cmd = cc + self._cc.compile_flag + cflags + ['-I', self._temp_dir] + ['-o', object_filename,
                                                                                      impl_filename]

        link_cmd = ld + self._cc.shared_lib_flag + lflags + ['-o', library_filename, object_filename]
        stdout_filename = filename_base + '.log'
        stderr_filename = filename_base + '.err'
        with open(stdout_filename, 'w') as stdout:
            with open(stderr_filename, 'w') as stderr:
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
        """Generate the source code the actual implementation.
        """

        d = {'UNIQUENAME': self._unique_name,
             'KERNEL': self._kernel_code,
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
# TOOLCHAIN TO COMPILE KERNEL AS LIBRARY
################################################################################################################

class SharedLib(GenericToolChain):
    """
    Generic base class to loop over all particles once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables and state variables.
    :arg bool runtime.DEBUG: Flag to enable runtime.DEBUG flags.
    """

    def __init__(self, kernel, particle_dat_dict):

        # Timers
        self.creation_timer = runtime.Timer(runtime.BUILD_TIMER, 2, start=True)
        """Timer that times the creation of the shared library if runtime.BUILD_TIMER.level > 2"""

        self.execute_timer = runtime.Timer(runtime.BUILD_TIMER, 2, start=False)
        """Timer that times the execution time of the shared library if runtime.BUILD_TIMER.level > 2"""

        self.execute_overhead_timer = runtime.Timer(runtime.BUILD_TIMER, 2, start=False)
        """Times the overhead required before the shared library is ran if runtime.BUILD_TIMER.level > 2. """

        self._compiler_set()
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
            if mpi.MPI_HANDLE.rank == 0:
                self._create_library()
            mpi.MPI_HANDLE.barrier()
        try:
            self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)
        except:
            load_library_exception(self._kernel.name, self._unique_name, type(self))

        self.creation_timer.stop("SharedLib creation timer " + str(self._kernel.name))

    def _compiler_set(self):
        self._cc = TMPCC

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

        for i, dat in enumerate(self._particle_dat_dict.items()):

            space = ' ' * 14
            argname = dat[0]  # +'_ext'
            loc_argname = argname  # dat[0]

            if issubclass(type(dat[1]), host.Array):
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

            if issubclass(type(dat[1]), host.Matrix):
                ncomp = dat[1].ncol
                s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';\n'
                s += space + loc_argname + ' = ' + argname + '+' + str(ncomp) + '*i;\n'

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

        void %(KERNEL_NAME)s_wrapper(%(ARGUMENTS)s);

        #endif
        '''

        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def execute(self, dat_dict=None, static_args=None):
        # Timing block 1
        self.execute_overhead_timer.start()


        """Allow alternative pointers"""
        if dat_dict is not None:

            for key in self._particle_dat_dict:
                self._particle_dat_dict[key] = dat_dict[key]

        args = []

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

        # Timing block 2
        self.execute_overhead_timer.pause()
        self.execute_timer.start()

        return_code = method(*args)

        # Timing block 3
        self.execute_timer.pause()
        self.execute_overhead_timer.start()

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
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
        #include \"%(UNIQUENAME)s.h\"

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


        for i, dat in enumerate(self._particle_dat_dict.items()):
            if type(dat[1]) is not tuple:
                argnames += host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + ','
            else:
                argnames += host.ctypes_map[dat[1][0].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + ','

        return argnames[:-1]