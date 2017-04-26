__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes
import os
import hashlib
import subprocess
import re

# package level imports
import config
import host
import runtime
import mpi
import opt


BUILD_PER_PROC = False

_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier

###############################################################################
# COMPILERS START
###############################################################################


TMPCC = config.COMPILERS[config.MAIN_CFG['cc-main'][1]]
TMPCC_OpenMP = config.COMPILERS[config.MAIN_CFG['cc-openmp'][1]]
MPI_CC = config.COMPILERS[config.MAIN_CFG['cc-mpi'][1]]


###############################################################################
# AUTOCODE TOOLS START
###############################################################################


build_dir = os.path.abspath(config.MAIN_CFG['build-dir'][1])

if not os.path.exists(build_dir) and _MPIRANK == 0:
    os.mkdir(build_dir)
_MPIBARRIER()



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
        f = open(os.path.join(runtime.BUILD_DIR, unique_name + '.err'), 'r')
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
                f = open(os.path.join(runtime.BUILD_DIR, unique_name + '.c'), 'r')
                code_str = f.read()
                f.close()
            except:
                print "Source file not read"

            code_str = code_str.split('\n')[max(0, err_line - 6):err_line + 1]
            code_str[-3] += "    <-------------"
            code_str = [x + "\n" for x in code_str]

            err_code = ''.join(code_str)
    print "Unique name", unique_name, "Rank", _MPIRANK

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




####################################
# Build Lib
####################################

def md5(string):
    """Create unique hex digest"""
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()

def source_write(header_code, src_code, name, extensions=('.h', '.cpp'), dst_dir=runtime.BUILD_DIR, CC=TMPCC):


    _filename = 'HOST_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) +
                           str(name))

    if BUILD_PER_PROC:
        _filename += '_' + str(_MPIRANK)

    _fh = open(os.path.join(dst_dir, _filename + extensions[0]), 'w')

    _fh.write('''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #define RESTRICT %(RESTRICT_FLAG)s
        ''' % {'UNIQUENAME':_filename, 'RESTRICT_FLAG':str(CC.restrict_keyword)})

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
    except Exception as e:
        print "build:load error. Could not load following library,", \
            str(filename)
        print e
        raise RuntimeError

def check_file_existance(abs_path=None):
    assert abs_path is not None, "build:check_file_existance error. " \
                                 "No absolute path passed."
    return os.path.exists(abs_path)

def simple_lib_creator(header_code, src_code, name, extensions=('.h', '.cpp'), dst_dir=runtime.BUILD_DIR, CC=TMPCC):
    if not os.path.exists(dst_dir) and _MPIRANK == 0:
        os.mkdir(dst_dir)


    _filename = 'HOST_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) +
                           str(name))

    if BUILD_PER_PROC:
        _filename += '_' + str(_MPIRANK)

    _lib_filename = os.path.join(dst_dir, _filename + '.so')

    if not check_file_existance(_lib_filename):

        if (_MPIRANK == 0)  or BUILD_PER_PROC:
            source_write(header_code, src_code, name, extensions=extensions, dst_dir=runtime.BUILD_DIR, CC=CC)
        build_lib(_filename, extensions=extensions, CC=CC, hash=False)

    return load(_lib_filename)

def build_lib(lib, extensions=('.h', '.cpp'), source_dir=runtime.BUILD_DIR,
              CC=TMPCC, dst_dir=runtime.BUILD_DIR, hash=True):


    if not BUILD_PER_PROC:
        _MPIBARRIER()

    with open(os.path.join(source_dir, lib + extensions[1]), "r") as fh:
        _code = fh.read()
        fh.close()
    with open(os.path.join(source_dir, lib + extensions[0]), "r") as fh:
        _code += fh.read()
        fh.close()

    if hash:
        _m = hashlib.md5()
        _m.update(_code)
        _m = '_' + _m.hexdigest()
    else:
        _m = ''

    _lib_filename = os.path.join(dst_dir, lib + str(_m) + '.so')


    if (_MPIRANK == 0) or BUILD_PER_PROC:
        if not os.path.exists(_lib_filename):

            _lib_src_filename = os.path.join(source_dir, lib + extensions[1])

            _c_cmd = [CC.binary] + [_lib_src_filename] + ['-o'] + \
                     [_lib_filename] + CC.c_flags  + CC.l_flags + \
                     ['-I' + str(runtime.LIB_DIR)] +\
                     ['-I' + str(source_dir)]

            if runtime.DEBUG > 0:
                _c_cmd += CC.dbg_flags
            if runtime.OPT > 0:
                _c_cmd += CC.opt_flags
            
            _c_cmd += CC.shared_lib_flag

            if runtime.VERBOSE > 2:
                print "Building", _lib_filename, _MPIRANK

            stdout_filename = os.path.join(dst_dir, lib + str(_m) + '.log')
            stderr_filename = os.path.join(dst_dir,  lib + str(_m) + '.err')
            try:
                with open(stdout_filename, 'w') as stdout:
                    with open(stderr_filename, 'w') as stderr:
                        stdout.write('#Compilation command:\n')
                        stdout.write(' '.join(_c_cmd))
                        stdout.write('\n\n')
                        p = subprocess.Popen(_c_cmd,
                                             stdout=stdout,
                                             stderr=stderr)
                        p.communicate()
            except Exception as e:
                print e
                raise RuntimeError('build error: library not built.')


    if not BUILD_PER_PROC:
        _MPIBARRIER()


    if not os.path.exists(_lib_filename):
        print "Critical build Error: Library not built,\n" + \
                   _lib_filename + "\n rank:", _MPIRANK

        if _MPIRANK == 0:
            with open(os.path.join(dst_dir, lib + str(_m) + '.err'), 'r') as stderr:
                print stderr.read()

        quit()

    return _lib_filename
















###############################################################################
# block of code class to be phased out
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

















