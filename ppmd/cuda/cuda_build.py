import hashlib
import os
import subprocess
import ctypes
import math

import cuda_runtime
from ppmd import build, mpi, runtime, host, access, pio

#####################################################################################
# NVCC Compiler
#####################################################################################

NVCC = build.Compiler(['nvcc_system_default'],
                      ['nvcc'],
                      ['-Xcompiler', '"-fPIC"'],
                      ['-lm'],
                      ['-O3', '--ptxas-options=-v -dlcm=ca', '--maxrregcount=64'],  # '-O3', '-Xptxas', '"-v"', '-lineinfo'
                      ['-G', '-g', '--source-in-ptx', '--ptxas-options=-v'],
                      ['-c', '-arch=sm_35', '-m64', '-lineinfo'],
                      ['-shared', '-Xcompiler', '"-fPIC"'],
                      '__restrict__')

#####################################################################################
# File writer helper function.
#####################################################################################


def md5(string):
    """Create unique hex digest"""
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()


def source_write(header_code, src_code, name, extensions=('.h', '.cu'), dst_dir=cuda_runtime.BUILD_DIR.dir):
    _filename = 'CUDA_' + str(name)
    _filename += '_' + md5(_filename + str(header_code) + str(src_code) + str(name))

    _fh = open(os.path.join(dst_dir, _filename + extensions[0]), 'w')
    _fh.write(str(header_code))
    _fh.close()

    _fh = open(os.path.join(dst_dir, _filename + extensions[1]), 'w')
    _fh.write('#include <' + _filename + extensions[0] + '>')
    _fh.write(str(src_code))
    _fh.close()

    return _filename, dst_dir

#####################################################################################
# build static libs
#####################################################################################

def build_static_libs(lib):
    return cuda_build_lib(lib, cuda_runtime.LIB_DIR.dir)


#####################################################################################
# build libs
#####################################################################################

def cuda_build_lib(lib, source_dir=cuda_runtime.BUILD_DIR.dir, CC=NVCC, dst_dir=cuda_runtime.BUILD_DIR.dir, hash=True):

    with open(source_dir + lib + ".cu", "r") as fh:
        _code = fh.read()
        fh.close()
    with open(source_dir + lib + ".h", "r") as fh:
        _code += fh.read()
        fh.close()

    if hash:
        _m = hashlib.md5()
        _m.update(_code)
        _m = '_' + _m.hexdigest()
    else:
        _m = ''

    _lib_filename = os.path.join(dst_dir, lib + str(_m) + '.so')

    if mpi.MPI_HANDLE.rank == 0:
        if not os.path.exists(_lib_filename):

            _lib_src_filename = source_dir + lib + '.cu'

            _c_cmd = CC.binary + [_lib_src_filename] + ['-o'] + [_lib_filename] + CC.c_flags \
                     + CC.l_flags + ['-I ' + str(cuda_runtime.LIB_DIR.dir)] + ['-I ' + str(source_dir)]
            if cuda_runtime.DEBUG.level > 0:
                _c_cmd += CC.dbg_flags
            else:
                _c_cmd += CC.opt_flags

            _c_cmd += CC.shared_lib_flag

            if cuda_runtime.VERBOSE.level > 2:
                print "Building", _lib_filename

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
                if cuda_runtime.ERROR_LEVEL.level > 2:
                    raise RuntimeError('gpucuda error: helper library not built.')
                elif cuda_runtime.VERBOSE.level > 2:
                    print "gpucuda warning: Shared library not built:", lib

    mpi.MPI_HANDLE.barrier()
    if not os.path.exists(_lib_filename):
        pio.pprint("Critical CUDA Error: Library not build, rank:", mpi.MPI_HANDLE.rank)
        quit()

    return _lib_filename

#####################################################################################
# build _base
#####################################################################################
















