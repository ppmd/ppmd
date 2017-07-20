from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes
import os
import hashlib
import subprocess

# package level imports
import ppmd.config
import ppmd.runtime
import ppmd.mpi


_MPIWORLD = ppmd.mpi.MPI.COMM_WORLD
_MPIRANK = ppmd.mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = ppmd.mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = ppmd.mpi.MPI.COMM_WORLD.Barrier

###############################################################################
# COMPILERS START
###############################################################################

TMPCC = ppmd.config.COMPILERS[ppmd.config.MAIN_CFG['cc-main'][1]]
TMPCC_OpenMP = ppmd.config.COMPILERS[ppmd.config.MAIN_CFG['cc-openmp'][1]]
MPI_CC = ppmd.config.COMPILERS[ppmd.config.MAIN_CFG['cc-mpi'][1]]

build_dir = os.path.abspath(ppmd.config.MAIN_CFG['build-dir'][1])

# make the tmp build directory
if not os.path.exists(build_dir) and _MPIRANK == 0:
    os.mkdir(build_dir)
_MPIBARRIER()


####################################
# Build Lib
####################################

def _md5(string):
    """Create unique hex digest"""
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()

def _source_write(header_code, src_code, name, extensions=('.h', '.cpp'), dst_dir=ppmd.runtime.BUILD_DIR, CC=TMPCC):


    _filename = 'HOST_' + str(name)
    _filename += '_' + _md5(_filename + str(header_code) + str(src_code) +
                            str(name))

    if ppmd.runtime.BUILD_PER_PROC:
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

def _load(filename):
    try:
        return ctypes.cdll.LoadLibrary(str(filename))
    except Exception as e:
        print("build:load error. Could not load following library,", \
            str(filename))
        print(e)
        raise RuntimeError

def _check_file_existance(abs_path=None):
    assert abs_path is not None, "build:check_file_existance error. " \
                                 "No absolute path passed."
    return os.path.exists(abs_path)

def simple_lib_creator(header_code, src_code, name, extensions=('.h', '.cpp'), dst_dir=ppmd.runtime.BUILD_DIR, CC=TMPCC):
    if not os.path.exists(dst_dir) and _MPIRANK == 0:
        os.mkdir(dst_dir)


    _filename = 'HOST_' + str(name)
    _filename += '_' + _md5(_filename + str(header_code) + str(src_code) +
                            str(name))

    if ppmd.runtime.BUILD_PER_PROC:
        _filename += '_' + str(_MPIRANK)

    _lib_filename = os.path.join(dst_dir, _filename + '.so')

    if not _check_file_existance(_lib_filename):

        if (_MPIRANK == 0)  or ppmd.runtime.BUILD_PER_PROC:
            _source_write(header_code, src_code, name, extensions=extensions, dst_dir=ppmd.runtime.BUILD_DIR, CC=CC)
        _build_lib(_filename, extensions=extensions, CC=CC, hash=False)

    return _load(_lib_filename)

def _build_lib(lib, extensions=('.h', '.cpp'), source_dir=ppmd.runtime.BUILD_DIR,
               CC=TMPCC, dst_dir=ppmd.runtime.BUILD_DIR, hash=True):

    if not ppmd.runtime.BUILD_PER_PROC:
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


    if (_MPIRANK == 0) or ppmd.runtime.BUILD_PER_PROC:
        if not os.path.exists(_lib_filename):

            _lib_src_filename = os.path.join(source_dir, lib + extensions[1])

            _c_cmd = CC.binary + [_lib_src_filename] + ['-o'] + \
                     [_lib_filename] + CC.c_flags  + CC.l_flags + \
                     ['-I' + str(ppmd.runtime.LIB_DIR)] + \
                     ['-I' + str(source_dir)]

            if ppmd.runtime.DEBUG > 0:
                _c_cmd += CC.dbg_flags
            if ppmd.runtime.OPT > 0:
                _c_cmd += CC.opt_flags
            
            _c_cmd += CC.shared_lib_flag

            #print("CCMD", _c_cmd)

            if ppmd.runtime.VERBOSE > 2:
                print("Building", _lib_filename, _MPIRANK)

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
                print(e)
                raise RuntimeError('build error: library not built.')


    if not ppmd.runtime.BUILD_PER_PROC:
        _MPIBARRIER()


    if not os.path.exists(_lib_filename):
        print("Critical build Error: Library not built,\n" + \
                   _lib_filename + "\n rank:", _MPIRANK)

        if _MPIRANK == 0:
            with open(os.path.join(dst_dir, lib + str(_m) + '.err'), 'r') as stderr:
                print(stderr.read())

        quit()

    return _lib_filename



























