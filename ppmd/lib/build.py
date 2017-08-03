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

    if ppmd.runtime.PY_MAJOR_VERSION > 2:
        string = string.encode('utf8')

    m = hashlib.md5()
    m.update(string)
    m = m.hexdigest()
    return m

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

def lib_from_source(
        base_filename,
        func_name,
        consts_dict=None,
        extensions=('.h', '.cpp'),
        cc=TMPCC
):
    """
    Compile and load a shared library from source files.
    :param base_filename: Base file name of source files e.g. "ABC" for 
    "ABC.cpp, ABC.h"
    :param func_name: Name of function in source files to load.
    :param consts_dict: Dictionary of form {'KEY': 'value'} that will be 
    applied to both code sources.
    :param extensions: Default ('.cpp', '.h') extensions to use with 
    base_filename.
    :param cc: Compiler to use, default set to default compiler.
    :return: compiled loaded library.
    """
    if consts_dict is None:
        consts_dict = {'RESTRICT': cc.restrict_keyword}
    else:
        assert type(consts_dict) is dict, "const_dict is not a dict"
        consts_dict['RESTRICT'] = cc.restrict_keyword

    with open(base_filename + extensions[0]) as fh:
        hsrc = fh.read() % consts_dict
    with open(base_filename + extensions[1]) as fh:
        src = fh.read() % consts_dict
    return simple_lib_creator(hsrc, src, func_name, extensions,
                              ppmd.runtime.BUILD_DIR, cc)[func_name]


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
        _build_lib(_filename, extensions=extensions, CC=CC)

    return _load(_lib_filename)

def _build_lib(lib, extensions=('.h', '.cpp'), source_dir=ppmd.runtime.BUILD_DIR,
               CC=TMPCC, dst_dir=ppmd.runtime.BUILD_DIR):

    _lib_filename = os.path.join(dst_dir, lib + '.so')

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

            stdout_filename = os.path.join(dst_dir, lib + '.log')
            stderr_filename = os.path.join(dst_dir,  lib + '.err')
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
            with open(os.path.join(dst_dir, lib + '.err'), 'r') as stderr:
                print(stderr.read())

        quit()

    return _lib_filename



























