import hashlib
import os
import subprocess
import ctypes

from ppmd import mpi, pio, config
import cuda_config
import cuda_runtime


NVCC = config.COMPILERS[cuda_config.CUDA_CFG['cc-main'][1]]

_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier

#####################################################################################
# File writer helper function.
#####################################################################################


def md5(string):
    """Create unique hex digest"""
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()


def source_write(header_code, src_code, name, extensions=('.h', '.cu'), dst_dir=cuda_runtime.BUILD_DIR):
    _filename = 'CUDA_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) + str(name))

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
    _fh.write('#include <' + _filename + extensions[0] + '>')
    _fh.write(str(src_code))
    _fh.close()

    return _filename, dst_dir


def load(filename):
    try:
        return ctypes.cdll.LoadLibrary(str(filename))
    except Exception as e:
        print "cuda_build:load error. Could not load following library,", str(filename)
        print e
        quit()

def check_file_existance(abs_path=None):
    assert abs_path is not None, "cuda_build:check_file_existance error. No absolute path passed."
    return os.path.exists(abs_path)

def simple_lib_creator(header_code, src_code, name, extensions=('.h', '.cu'), dst_dir=cuda_runtime.BUILD_DIR):
    _filename = 'CUDA_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) + str(name))
    _lib_filename = os.path.join(dst_dir, _filename + '.so')

    if not check_file_existance(_lib_filename):
        source_write(header_code, src_code, name, extensions=('.h', '.cu'), dst_dir=cuda_runtime.BUILD_DIR)
        cuda_build_lib(_filename, hash=False)

    return load(_lib_filename)


#####################################################################################
# build static libs
#####################################################################################

def build_static_libs(lib):
    return cuda_build_lib(lib, cuda_runtime.LIB_DIR)


#####################################################################################
# build libs
#####################################################################################

def cuda_build_lib(lib, source_dir=cuda_runtime.BUILD_DIR, CC=NVCC, dst_dir=cuda_runtime.BUILD_DIR, hash=True):

    if hash:
        with open(os.path.join(source_dir, lib + ".cu"), "r") as fh:
            _code = fh.read()
            fh.close()
        with open(os.path.join(source_dir, lib + ".h"), "r") as fh:
            _code += fh.read()
            fh.close()

        _m = hashlib.md5()
        _m.update(_code)
        _m = '_' + _m.hexdigest()
    else:
        _m = ''

    _lib_filename = os.path.join(dst_dir, lib + str(_m) + '.so')

    if _MPIRANK == 0:
        if not os.path.exists(_lib_filename):

            _lib_src_filename = os.path.join(source_dir, lib + '.cu')

            _c_cmd = [CC.binary] + [_lib_src_filename] + ['-o'] + [_lib_filename] + CC.c_flags \
                     + CC.l_flags + ['-I ' + str(cuda_runtime.LIB_DIR)] + ['-I ' + str(source_dir)]
            if cuda_runtime.DEBUG > 0:
                _c_cmd += CC.dbg_flags

            if cuda_runtime.OPT > 0:
                _c_cmd += CC.opt_flags

            _c_cmd += CC.shared_lib_flag

            if cuda_runtime.VERBOSE > 1:
                print "Building", _lib_filename

            stdout_filename = os.path.join(dst_dir, lib + str(_m) + '.log')
            stderr_filename = os.path.join(dst_dir, lib + str(_m) + '.err')
            try:
                with open(stdout_filename, 'w') as stdout:
                    with open(stderr_filename, 'w') as stderr:
                        stdout.write('# Compilation command:\n')
                        stdout.write('# ' + str(_c_cmd) + '\n')
                        stdout.write('# \n')

                        stdout.write(' '.join(_c_cmd))
                        stdout.write('\n\n')
                        p = subprocess.Popen(_c_cmd,
                                             stdout=stdout,
                                             stderr=stderr)
                        p.communicate()
            except:
                if cuda_runtime.ERROR_LEVEL > 2:
                    raise RuntimeError('cuda_build error: Shared library not built.')
                elif cuda_runtime.VERBOSE > 2:
                    print "cuda_build warning: Shared library not built:", lib

    _MPIBARRIER()

    if not os.path.exists(_lib_filename):
        pio.pprint("Critical CUDA Error: Library not built, " + str(lib) + ", rank:", _MPIRANK)

        if _MPIRANK == 0:
            with open(os.path.join(dst_dir, lib + str(_m) + '.err'), 'r') as stderr:
                print stderr.read()

        _MPIBARRIER()


        quit()

    return _lib_filename


#####################################################################################
# block of code class
#####################################################################################

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







