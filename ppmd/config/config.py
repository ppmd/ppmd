"""
Configuration handling for package
"""
from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import os
import configparser as ConfigParser
import mpi4py
import shlex
import hashlib

# package level imports
from ppmd.lib import compiler
from glob import glob

def str_to_bool(s="0"):
    return bool(int(s))

COMPILERS = dict()
MAIN_CFG = dict()


# defaults and type defs for main options
MAIN_CFG['opt-level'] = (int, 1)
MAIN_CFG['verbose-level'] = (int, 0)
MAIN_CFG['timer-level'] = (int, 1)
MAIN_CFG['build-timer-level'] = (int, 0)
MAIN_CFG['error-level'] = (int, 3)

build_dir = '/tmp/build'
if 'PPMD_BUILD_DIR' in os.environ:
    build_dir = os.environ['PPMD_BUILD_DIR']

if 'PPMD_CC_MAIN' in os.environ:
    cc_main = os.environ['PPMD_CC_MAIN']
else:
    cc_main = 'GCC'

if 'PPMD_CC_OMP' in os.environ:
    cc_omp = os.environ['PPMD_CC_OMP']
else:
    cc_omp = 'GCC'

if 'PPMD_EXTRA_COMPILERS' in os.environ:
    extra_compiler_dir = os.path.abspath(os.environ['PPMD_EXTRA_COMPILERS'])
else:
    extra_compiler_dir = None

if 'PPMD_LOCAL_LIB_DIR' in os.environ:
    local_lib_dir = os.path.abspath(os.environ['PPMD_LOCAL_LIB_DIR'])
else:
    local_lib_dir = build_dir

if 'PPMD_ENABLE_DEBUG' in os.environ:
    MAIN_CFG['debug-level'] = (int, 1)
else:
    MAIN_CFG['debug-level'] = (int, 0)


MAIN_CFG['build-dir'] = (str, build_dir)
MAIN_CFG['cc-main'] = (str, cc_main)
MAIN_CFG['cc-openmp'] = (str, cc_omp)
MAIN_CFG['cc-mpi'] = (str, 'MPI4PY')
MAIN_CFG['local_lib_dir'] = (str, local_lib_dir)


def load_config(dir=None):
    if dir is None:
        CFG_DIR = os.path.dirname(os.path.realpath(__file__))
    else:
        CFG_DIR = os.path.abspath(dir)

    """
    # parse main options
    main_parser = ConfigParser.ConfigParser(os.environ)
    main_parser.read(os.path.join(CFG_DIR, 'default.cfg'))
    for key in MAIN_CFG:
        try:
            t = MAIN_CFG[key][0]
            MAIN_CFG[key] = (t, t(main_parser.get('ppmd', key)))
        except ConfigParser.InterpolationError:
            pass
        except ConfigParser.NoOptionError:
            pass
    """

    CC_KEYS = (
                'name',
                'binary',
                'compile-flags',
                'link-flags',
                'opt-flags',
                'debug-flags',
                'compile-object-flag',
                'shared-object-flag',
                'restrict-keyword'
              )


    # parse all config files in the compilers dir.
    cc_parser = ConfigParser.ConfigParser()
 
    compiler_dirs = glob(os.path.join(CFG_DIR, 'compilers/*.cfg'))
    
    if extra_compiler_dir is not None:
        compiler_dirs += glob(os.path.join(extra_compiler_dir, '*.cfg'))

    for cc_cfg in compiler_dirs:
        #if not cc_cfg.endswith('.cfg'):
        #    continue
        
        with open(os.path.join(os.path.join(CFG_DIR, 'compilers'), cc_cfg)) as fh:
            cnts = fh.read()
        
        cc_parser.read_string(cnts)

        args = []
        for key in CC_KEYS:
            try:
                getval = cc_parser.get('compiler', key)
            except ConfigParser.InterpolationError:
                pass
            except ConfigParser.NoOptionError:
                pass

            if key in ('name', 'restrict-keyword'):
                args.append(getval)
            else:
                args.append(shlex.split(getval))

        h = hashlib.md5()
        h.update(cnts.encode('UTF-8'))
        args.append(h.hexdigest())

        COMPILERS[args[0]] = compiler.Compiler(*args)
    

    mpi4py_config = mpi4py.get_config()
    if 'mpicxx' in mpi4py_config.keys():
        mpi_cc = mpi4py_config['mpicxx']
    elif 'mpicc' in mpi4py_config.keys():
        mpi_cc = mpi4py_config['mpicc']
    else:
        raise RuntimeError('Cannot find MPI compiler used to build mpi4py.')
    
    h = hashlib.md5()
    for keyx in mpi4py_config:
        h.update(str(keyx + mpi4py_config[keyx]).encode('UTF-8'))

    # create the mpi4py compiler
    tm = COMPILERS['MPI4PY']
    COMPILERS['MPI4PY'] = compiler.Compiler(
        name='MPI4PY',
        binary=[mpi_cc,],
        c_flags=tm.c_flags,
        l_flags=tm.l_flags,
        opt_flags=tm.opt_flags,
        dbg_flags=tm.dbg_flags,
        compile_flag=tm.compile_flag,
        shared_lib_flag=tm.shared_lib_flag,
        restrict_keyword=tm.restrict_keyword,
        cfg_hash=h.hexdigest()
    )


    assert MAIN_CFG['cc-main'][1] in COMPILERS.keys(), "cc-main compiler config not found"
    assert MAIN_CFG['cc-openmp'][1] in COMPILERS.keys(), "cc-openmp compiler config not found"
    assert MAIN_CFG['cc-mpi'][1] in COMPILERS.keys(), "cc-mpi compiler config not found"


load_config()
