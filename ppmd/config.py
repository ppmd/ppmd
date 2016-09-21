"""
Configuration handling for package
"""

# system level imports
import os
import ConfigParser
from mpi4py import MPI

# package level imports
import compiler

def str_to_bool(s="0"):
    return bool(int(s))

COMPILERS = dict()
MAIN_CFG = dict()


# defaults and type defs for main options
MAIN_CFG['opt-level'] = (int, 1)
MAIN_CFG['debug-level'] = (int, 0)
MAIN_CFG['verbose-level'] = (int, 0)
MAIN_CFG['timer-level'] = (int, 1)
MAIN_CFG['build-timer-level'] = (int, 0)
MAIN_CFG['error-level'] = (int, 3)
MAIN_CFG['build-dir'] = (str, os.path.join(os.getcwd(), 'build'))
MAIN_CFG['cc-main'] = (str, 'ICC')
MAIN_CFG['cc-openmp'] = (str, 'ICC')
MAIN_CFG['cc-mpi'] = (str, 'OPENMPI')



def load_config(dir=None):
    if dir is None:
        CFG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_dir')
    else:
        CFG_DIR = os.path.abspath(dir)


    # parse main options
    main_parser = ConfigParser.SafeConfigParser(os.environ)
    main_parser.read(os.path.join(CFG_DIR, 'default.cfg'))
    for key in MAIN_CFG:
        try:
            t = MAIN_CFG[key][0]
            MAIN_CFG[key] = (t, t(main_parser.get('ppmd', key)))
        except ConfigParser.InterpolationError:
            pass
        except ConfigParser.NoOptionError:
            pass


    build_dir = os.path.abspath(MAIN_CFG['build-dir'][1])
    if not os.path.exists(build_dir) and MPI.COMM_WORLD.rank == 0:
        os.mkdir(build_dir)
    MPI.COMM_WORLD.Barrier()






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
    cc_parser = ConfigParser.SafeConfigParser()
    for cc_cfg in os.listdir(os.path.join(CFG_DIR, 'compilers')):
        cc_parser.read(os.path.join(os.path.join(CFG_DIR, 'compilers'), cc_cfg))
        args = []
        for key in CC_KEYS:
            try:
                args.append(cc_parser.get('compiler', key))
            except ConfigParser.InterpolationError:
                pass
            except ConfigParser.NoOptionError:
                pass

        COMPILERS[args[0]] = compiler.Compiler(*args)


    assert MAIN_CFG['cc-main'][1] in COMPILERS.keys(), "cc-main compiler config not found"
    assert MAIN_CFG['cc-openmp'][1] in COMPILERS.keys(), "cc-openmp compiler config not found"
    assert MAIN_CFG['cc-mpi'][1] in COMPILERS.keys(), "cc-mpi compiler config not found"




load_config()
