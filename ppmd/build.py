import sys

import host
import ctypes
import os
import hashlib
import subprocess
import re
import pio
import runtime
import mpi


_PER_PROC = False

###############################################################################
# COMPILERS START
###############################################################################


class Compiler(object):
    """
    Container to define different compilers.
    
    :arg str name: Compiler name, referance only.
    :arg str binary: Name(+path) of Compiler binary.
    :arg list c_flags: List of compile flags as strings.
    :arg list l_flags: List of link flags as strings.
    :arg list opt_flags: List of optimisation flags.
    :arg list dbg_flags: List of runtime.DEBUG flags as strings.
    :arg list compile_flag: List of compile flag as single string (eg ['-c']
    for gcc).
    :arg list shared_lib_flag: List of flags as strings to link as shared
    library.
    :arg string restrict_keyword: keyword to use for non aliased pointers
    """

    def __init__(self, name, binary, c_flags, l_flags, opt_flags, dbg_flags,
                 compile_flag, shared_lib_flag, restrict_keyword=''):

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



GCC = Compiler(['GCC'],
               ['g++'],
               ['-fPIC', '-std=c++0x'],
               ['-lm'],
               ['-O3', '-march=native', '-m64', '-ftree-vectorizer-verbose=5', '-fassociative-math'],
               ['-g'],
               ['-c'],
               ['-shared'],
               '__restrict__')





# Define system gcc version as OpenMP Compiler.
GCC_OpenMP = Compiler(['GCC'],
                      ['g++'],
                      ['-fopenmp', '-fPIC', '-std=c++0x'],
                      ['-lgomp', '-lrt', '-Wall'],
                      ['-O3', '-march=native', '-m64','-ftree-vectorizer-verbose=5'],
                      ['-g'],
                      ['-c', '-Wall'],
                      ['-shared'],
                      '__restrict__')


# Define system icc version as Compiler.
ICC = Compiler(['ICC'],
               ['icc'],
               ['-fpic', '-std=c++0x'],
               ['-lm'],
               ['-O3', '-xHost', '-restrict', '-m64', '-qopt-report=4'],
               ['-g'],
               ['-c'],
               ['-shared'],
               'restrict')

try:
    ICC_MPI = Compiler(['ICC'],
                       ['icc'],
                       ['-fpic', '-std=c++0x'],
                       ['-lm'],
                       ['-O3', '-xHost', '-restrict', '-m64', '-qopt-report=4', '-I' + os.environ["MPI_INCLUDE_DIR"]],
                       ['-lmpi'],
                       ['-c'],
                       ['-shared'],
                       'restrict')
except:
    pass

try:
    GCC_MPI = Compiler(['GCC'],
               ['mpic++'],
               ['-fPIC', '-std=c++0x'],
               ['-lm'],
               ['-O3', '-march=native', '-m64', '-ftree-vectorizer-verbose=5', '-fassociative-math', '-I' + os.environ['MPI_HOME'] + '/include'],
               ['-g'],
               ['-c'],
               ['-shared'],
               '__restrict__')
except:
    pass
# Define system icc version as OpenMP Compiler.
ICC_OpenMP = Compiler(['ICC'],
                      ['icc'],
                      ['-fpic', '-openmp', '-std=c++0x'],
                      ['-openmp', '-lgomp', '-lpthread', '-lc', '-lrt'],
                      ['-O3', '-xHost', '-restrict', '-m64', '-qopt-report=4'],
                      ['-g'],
                      ['-c'],
                      ['-shared'],
                      'restrict')


# Temporary Compiler flag
ICC_LIST = ['mapc-4044']#, 'itd-ngpu-01', 'itd-ngpu-02']

if os.uname()[1] in ICC_LIST:
    TMPCC = ICC
    TMPCC_OpenMP = ICC_OpenMP
    MPI_CC = ICC_MPI
else:
    TMPCC = GCC
    TMPCC_OpenMP = GCC_OpenMP
    MPI_CC = GCC_MPI


###############################################################################
# AUTOCODE TOOLS START
###############################################################################


def load_library_exception(kernel_name='None supplied',
                           unique_name='None supplied',
                           looping_type='None supplied'):

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
                       "################################################### \n"
                       "\t \t \t ERROR \n"
                       "################################################### \n"
                       "kernel name: " + str(kernel_name) + "\n"
                       "--------------------------------------------------- \n"
                       "looping class: " + str(looping_type) + "\n"
                       "--------------------------------------------------- \n"
                       "Compile/link error message: \n \n" +
                       str(err_msg) + "\n"
                       "--------------------------------------------------- \n"
                       "Error location attempt: \n \n" +
                       str(err_code) + "\n \n"
                       "################################################### \n"
                       )


def loop_unroll(str_start, i, j, step, str_end=None, key=None):
    """
    Function to create unrolled loops in python source.
    Potentialy add auto padding here?
    
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




###############################################################################
# TOOLCHAIN TO COMPILE KERNEL AS LIBRARY
###############################################################################

class SharedLib(object):
    """
    Generic base class to loop over all particles once.
    
    :arg int n: Number of elements to loop over.
    :arg kernel kernel:  Kernel to apply at each element.
    :arg dict particle_dat_dict: Dictonary storing map between kernel variables
    and state variables.
    :arg bool runtime.DEBUG: Flag to enable runtime.DEBUG flags.
    """

    def __init__(self, kernel, particle_dat_dict, openmp=False):

        # Timers
        self.creation_timer = runtime.Timer(runtime.BUILD_TIMER, 2, start=True)
        """Timer that times the creation of the shared library if
        runtime.BUILD_TIMER.level > 2"""

        self.execute_timer = runtime.Timer(runtime.TIMER, 0, start=False)
        """Timer that times the execution time of the shared library if
        runtime.BUILD_TIMER.level > 2"""

        self.execute_overhead_timer = runtime.Timer(runtime.BUILD_TIMER, 2,
                                                    start=False)
        """Times the overhead required before the shared library is ran if
        runtime.BUILD_TIMER.level > 2. """

        self._omp = openmp

        self._compiler_set()
        self._temp_dir = runtime.BUILD_DIR.dir

        self._kernel = kernel

        self._particle_dat_dict = particle_dat_dict
        self._nargs = len(self._particle_dat_dict)

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

        for i, dat in enumerate(self._particle_dat_dict.items()):

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
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
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
        for i, dat in enumerate(self._particle_dat_dict.items()):
            # dat[0] is always the name, even with access descriptiors.
            argnames += dat[0] + ','
        return argnames[:-1]

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

        d = {'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'LOC_ARGUMENTS': self._loc_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations()}

        return self._code % d




####################################
# Build Lib
####################################

def md5(string):
    """Create unique hex digest"""
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()

def source_write(header_code, src_code, name, extensions=('.h', '.cpp'), dst_dir=runtime.BUILD_DIR.dir):


    _filename = 'HOST_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) +
                           str(name))

    if _PER_PROC:
        _filename += '_' + str(mpi.MPI_HANDLE.rank)

    _fh = open(os.path.join(dst_dir, _filename + extensions[0]), 'w')
    _fh.write('''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        ''' % {'UNIQUENAME':_filename})

    _fh.write(str(header_code))

    _fh.write('''
        #endif
        ''' % {'UNIQUENAME':_filename})

    _fh.close()

    _fh = open(os.path.join(dst_dir, _filename + extensions[1]), 'w')
    _fh.write('#include <' + _filename + extensions[0] + '>\n\n')
    _fh.write(str(src_code))
    _fh.close()

    return _filename, dst_dir

def load(filename):
    try:
        return ctypes.cdll.LoadLibrary(str(filename))
    except:
        print "build:load error. Could not load following library,", \
            str(filename)
        quit()

def check_file_existance(abs_path=None):
    assert abs_path is not None, "build:check_file_existance error. " \
                                 "No absolute path passed."
    return os.path.exists(abs_path)

def simple_lib_creator(header_code, src_code, name, extensions=('.h', '.cpp'), dst_dir=runtime.BUILD_DIR.dir, CC=TMPCC):
    if not os.path.exists(dst_dir) and mpi.MPI_HANDLE.rank == 0:
        os.mkdir(dst_dir)


    _filename = 'HOST_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) +
                           str(name))

    if _PER_PROC:
        _filename += '_' + str(mpi.MPI_HANDLE.rank)

    _lib_filename = os.path.join(dst_dir, _filename + '.so')

    if not check_file_existance(_lib_filename):

        if (mpi.MPI_HANDLE.rank == 0)  or _PER_PROC:
            source_write(header_code, src_code, name, extensions=extensions, dst_dir=runtime.BUILD_DIR.dir)
        build_lib(_filename, extensions=extensions, CC=CC, hash=False)

    return load(_lib_filename)

def build_lib(lib, extensions=('.h', '.cpp'), source_dir=runtime.BUILD_DIR.dir,
              CC=TMPCC, dst_dir=runtime.BUILD_DIR.dir, hash=True):

    mpi.MPI_HANDLE.barrier()

    with open(source_dir + lib + extensions[1], "r") as fh:
        _code = fh.read()
        fh.close()
    with open(source_dir + lib + extensions[0], "r") as fh:
        _code += fh.read()
        fh.close()

    if hash:
        _m = hashlib.md5()
        _m.update(_code)
        _m = '_' + _m.hexdigest()
    else:
        _m = ''

    _lib_filename = os.path.join(dst_dir, lib + str(_m) + '.so')

    #print mpi.MPI_HANDLE.rank, "building", _lib_filename

    if (mpi.MPI_HANDLE.rank == 0) or _PER_PROC:
        if not os.path.exists(_lib_filename):

            _lib_src_filename = source_dir + lib + extensions[1]

            _c_cmd = CC.binary + [_lib_src_filename] + ['-o'] + \
                     [_lib_filename] + CC.c_flags  + CC.l_flags + \
                     ['-I' + str(runtime.LIB_DIR.dir)] + ['-I' + str(source_dir)]
            if runtime.DEBUG.level > 0:
                _c_cmd += CC.dbg_flags
            if runtime.OPT.level > 0:
                _c_cmd += CC.opt_flags

            _c_cmd += CC.shared_lib_flag

            if runtime.VERBOSE.level > 2:
                print "Building", _lib_filename, mpi.MPI_HANDLE.rank

            stdout_filename = dst_dir + lib + str(_m) + '.log'
            stderr_filename = dst_dir + lib + str(_m) + '.err'
            try:
                with open(stdout_filename, 'w') as stdout:
                    with open(stderr_filename, 'w') as stderr:
                        stdout.write('Compilation command:\n')
                        stdout.write(' '.join(_c_cmd))
                        stdout.write('\n\n')
                        p = subprocess.Popen(_c_cmd,
                                             stdout=stdout,
                                             stderr=stderr)
                        p.communicate()
            except:
                if runtime.ERROR_LEVEL.level > 2:
                    raise RuntimeError('build error: library not built.')
                elif runtime.VERBOSE.level > 2:
                    print "build error: Shared library not built:", lib

    #print "before barrier", mpi.MPI_HANDLE.rank
    #sys.stdout.flush()

    mpi.MPI_HANDLE.barrier()


    #print "after barrier", mpi.MPI_HANDLE.rank
    #sys.stdout.flush()

    #mpi.MPI_HANDLE.barrier()


    if not os.path.exists(_lib_filename):
        pio.pprint("Critical build Error: Library not built,\n" +
                   _lib_filename + "\n rank:", mpi.MPI_HANDLE.rank)

        if mpi.MPI_HANDLE.rank == 0:
            with open(dst_dir + lib + str(_m) + '.err', 'r') as stderr:
                print stderr.read()

        quit()

    return _lib_filename



###############################################################################
# block of code class
###############################################################################

class Code(object):
    def __init__(self, init=''):
        self._c = str(init)

    @property
    def string(self):
        return self._c

    def add_line(self, line=''):
        self._c += '\n' + str(line)

    def add(self, code=''):
        self._c += str(code)

    def __iadd__(self, other):
        self.add(code=str(other))
        return self

    def __str__(self):
        return str(self._c)

    def __add__(self, other):
        return Code(self.string + str(other))

















