from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes
import os
import stat
import hashlib
import subprocess
import sys
from pytools.prefork import call_capture_output
import numpy as np
from tempfile import TemporaryDirectory


# package level imports
from ppmd import config, runtime, mpi, opt
import ppmd.lib


_MPIRANK = ppmd.mpi.MPI.COMM_WORLD.Get_rank()
_MPIWORLD = ppmd.mpi.MPI.COMM_WORLD
_MPIBARRIER = ppmd.mpi.MPI.COMM_WORLD.Barrier



TMPCC = ppmd.config.COMPILERS[ppmd.config.MAIN_CFG['cc-main'][1]]
TMPCC_OpenMP = ppmd.config.COMPILERS[ppmd.config.MAIN_CFG['cc-openmp'][1]]
MPI_CC = ppmd.config.COMPILERS[ppmd.config.MAIN_CFG['cc-mpi'][1]]

build_dir = os.path.abspath(ppmd.config.MAIN_CFG['build-dir'][1])



# make the tmp build directory
if not os.path.exists(build_dir) and _MPIRANK == 0:
    os.mkdir(build_dir)

_MPIBARRIER()

_lldir = ppmd.config.MAIN_CFG['local_lib_dir'][1]
LOCAL_LIB_DIR = TemporaryDirectory(prefix='ppmd_lld_', dir=_lldir)


LOADED_LIBS = []


def _read_lib_as_bytes(directory, filename):
    with open(os.path.join(directory, filename), 'rb') as fh:
        f = fh.read()
    return f



def _md5(string):
    """Create unique hex digest"""

    if ppmd.runtime.PY_MAJOR_VERSION > 2:
        string = string.encode('utf8')

    m = hashlib.md5()
    m.update(string)
    m = m.hexdigest()
    return m

def lib_from_file_source(
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
        s = fh.read()
        src = s % consts_dict
    return simple_lib_creator(hsrc, src, func_name, extensions,
        ppmd.runtime.BUILD_DIR, cc)

def _source_write(header_code, src_code, filename, extensions, dst_dir, CC):
    with open(os.path.join(dst_dir, filename + extensions[0]), 'w') as fh:
        fh.write('''
            #ifndef %(UNIQUENAME)s_H
            #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
            #define RESTRICT %(RESTRICT_FLAG)s
            #include <cstdint>
            %(HEADER_CODE)s
            
            #endif
            ''' % {
            'UNIQUENAME':filename,
            'RESTRICT_FLAG':str(CC.restrict_keyword),
            'HEADER_CODE': str(header_code)
        })

    with open(os.path.join(dst_dir, filename + extensions[1]), 'w') as fh:
        fh.write('#include <' + filename + extensions[0] + '>\n\n')
        fh.write(str(src_code))
    return filename, dst_dir

def _load(filename):
    try:
        lib = ctypes.cdll.LoadLibrary(str(filename))
        LOADED_LIBS.append(str(filename[:-3]))
        return lib
    except Exception as e:
        print("build:load error. Could not load following library,", \
            str(filename))
        ppmd.abort(e)

def _check_path_exists(abs_path):
    return os.path.exists(abs_path)

_load_timer = opt.Timer()

def simple_lib_creator(
        header_code, src_code, name='', extensions=('.h', '.cpp'),
        dst_dir=ppmd.runtime.BUILD_DIR, CC=TMPCC, prefix='HOST',
        inc_dirs=(runtime.LIB_DIR,)
):

    # make build dir
    if not os.path.exists(dst_dir) and _MPIRANK == 0:
        os.mkdir(dst_dir)

    # create a base filename for the library
    _filename = prefix + '_' + str(name)
    _filename += '_' + _md5(_filename + str(header_code) + str(src_code) +
                            str(name) + str(CC.hash) + str(ppmd.runtime.DEBUG))

    if ppmd.runtime.BUILD_PER_PROC:
        _filename += '_' + str(_MPIRANK)

    _lib_filename = os.path.join(dst_dir, _filename + '.so')
    
    _build_needed = not _check_path_exists(_lib_filename)

    if not ppmd.runtime.BUILD_PER_PROC:

        var = int(hashlib.md5(_filename.encode('utf-8')).hexdigest()[:7], 16)
        var0 = np.array([int(_build_needed), var])
        _MPIWORLD.Bcast(var0, root=0)
        if var0[1] != var:
            ppmd.abort('Consensus not reached on filename:' + \
                _filename)

        if var0[0] != int(_build_needed):
            ppmd.abort('Consensus not reached on build needed:' + \
                _filename)

    if _build_needed:

        # need all ranks to recognise file does not exist if not build per proc
        # before rank 0 starts to build it
        if not ppmd.runtime.BUILD_PER_PROC:
            _MPIBARRIER()

        if (_MPIRANK == 0)  or ppmd.runtime.BUILD_PER_PROC:
            _source_write(header_code, src_code, _filename,
                          extensions=extensions,
                          dst_dir=dst_dir, CC=CC)

            build_lib(_filename, extensions=extensions, source_dir=dst_dir,
                      CC=CC, dst_dir=dst_dir, inc_dirs=inc_dirs)
        if not ppmd.runtime.BUILD_PER_PROC:
            _MPIBARRIER()
    
    if _MPIRANK == 0:
        fb = _read_lib_as_bytes(dst_dir, _lib_filename)
        a=len(fb)
    else:
        a=-1
    a = np.array(a)

    _MPIWORLD.Bcast(a)
    assert a > 0

    if _MPIRANK != 0:
        fb = bytearray(a)

    _MPIWORLD.Bcast(fb)
    print(_MPIRANK, len(fb))
    

    local_filename = os.path.join(LOCAL_LIB_DIR.name, _filename + '.so')
    print(_lib_filename)
    print(local_filename)

    with open(local_filename, 'wb') as fh:
        fh.write(fb)

    os.chmod(local_filename, stat.S_IRWXU)


    _load_timer.start()
    # lib = _load(_lib_filename)
    lib = _load(local_filename)
    _load_timer.pause()

    opt.PROFILE['Lib-Load'] = _load_timer.time()

    return lib

_build_timer = opt.Timer()

def _print_file_if_exists(filename):
    if os.path.exists(filename):
        with open(filename) as fh:
            print(fh.read())


def build_lib(lib, extensions, source_dir, CC, dst_dir, inc_dirs):
    _build_timer.start()

    _lib_filename = os.path.join(dst_dir, lib + '.so')
    _lib_src_filename = os.path.join(source_dir, lib + extensions[1])

    _c_cmd = CC.binary + [_lib_src_filename] + ['-o'] + \
             [_lib_filename] + CC.c_flags  + CC.l_flags + \
             ['-I' + str(d) for d in inc_dirs] + \
             ['-I' + str(source_dir)]
    
    if ppmd.runtime.DEBUG > 0:
        _c_cmd += CC.dbg_flags
    else:
        _c_cmd += CC.opt_flags

    _c_cmd += CC.shared_lib_flag

    stdout_filename = os.path.join(dst_dir, lib + '.log')
    stderr_filename = os.path.join(dst_dir,  lib + '.err')
    with open(stdout_filename, 'w') as stdout_fh:
        with open(stderr_filename, 'w') as stderr_fh:
            stdout_fh.write('# Compilation command:\n')
            stdout_fh.write(' '.join(_c_cmd))
            stdout_fh.write('\n\n')
            stdout_fh.flush()
            result, stdout, stderr = call_capture_output(_c_cmd, error_on_nonzero=False)

            stdout = stdout.decode(sys.stdout.encoding)
            stderr = stderr.decode(sys.stdout.encoding)

            stderr_fh.write(str(result) + '\n')
            stderr_fh.write(stdout)
            stderr_fh.write(stderr)
            stderr_fh.flush()
            if result != 0:
                print("\n---- COMPILER OUTPUT START ----")
                print(stdout)
                print(stderr)
                print("----- COMPILER OUTPUT END -----")
                raise RuntimeError('PPMD build error: library not built.')

    # Check library exists in the file system
    if not os.path.exists(_lib_filename):
        print("Critical build Error: Library not found,\n" + \
                   _lib_filename + "\n rank:", _MPIRANK)
        raise RuntimeError('compiler call did not error, but no binary found')

    _build_timer.pause()
    opt.PROFILE['Build:' + CC.binary[0] + ':'] = (_build_timer.time())
    return _lib_filename



























