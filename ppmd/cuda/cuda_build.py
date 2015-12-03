import hashlib
import os
import subprocess
import cuda_runtime
from ppmd import build, mpi

#####################################################################################
# NVCC Compiler
#####################################################################################

NVCC = build.Compiler(['nvcc_system_default'],
                      ['nvcc'],
                      ['-Xcompiler', '"-fPIC"'],
                      ['-lm'],
                      ['-O3', '--ptxas-options=-v -dlcm=ca', '--maxrregcount=64'],  # '-O3', '-Xptxas', '"-v"', '-lineinfo'
                      ['-G', '-g', '--source-in-ptx', '--ptxas-options="-v -dlcm=ca"'],
                      ['-c', '-arch=sm_35', '-m64', '-lineinfo'],
                      ['-shared', '-Xcompiler', '"-fPIC"'],
                      '__restrict__')


#####################################################################################
# build static libs
#####################################################################################

def build_static_libs(lib):

    with open(cuda_runtime.LIB_DIR.dir + lib + ".cu", "r") as fh:
        _code = fh.read()
        fh.close()
    with open(cuda_runtime.LIB_DIR.dir + lib + ".h", "r") as fh:
        _code += fh.read()
        fh.close()

    _m = hashlib.md5()
    _m.update(_code)
    _m = _m.hexdigest()

    _lib_filename = os.path.join(cuda_runtime.BUILD_DIR.dir, lib + '_' +str(_m) +'.so')

    if mpi.MPI_HANDLE.rank == 0:
        if not os.path.exists(_lib_filename):


            _lib_src_filename = cuda_runtime.LIB_DIR.dir + lib + '.cu'

            _c_cmd = NVCC.binary + [_lib_src_filename] + ['-o'] + [_lib_filename] + NVCC.c_flags + NVCC.l_flags
            if cuda_runtime.DEBUG.level > 0:
                _c_cmd += NVCC.dbg_flags
            else:
                _c_cmd += NVCC.opt_flags

            _c_cmd += NVCC.shared_lib_flag

            if cuda_runtime.VERBOSE.level > 2:
                print "Building", _lib_filename

            stdout_filename = cuda_runtime.BUILD_DIR.dir + lib + '_' +str(_m) + '.log'
            stderr_filename = cuda_runtime.BUILD_DIR.dir + lib + '_' +str(_m) + '.err'
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

    return _lib_filename

