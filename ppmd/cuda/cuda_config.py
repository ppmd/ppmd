from __future__ import print_function, division, absolute_import
"""
Configuration handling for package
"""

# system level imports
import os
import configparser as ConfigParser


# package level imports
import ppmd.config as config


CUDA_CFG = dict()

# defaults for cuda
CUDA_CFG['opt-level'] = (int, 1)
CUDA_CFG['debug-level'] = (int, 0)
CUDA_CFG['verbose-level'] = (int, 0)
CUDA_CFG['timer-level'] = (int, 1)
CUDA_CFG['build-timer-level'] = (int, 0)
CUDA_CFG['error-level'] = (int, 3)
CUDA_CFG['enable-cuda'] = (config.str_to_bool, False)
CUDA_CFG['cc-main'] = (str, 'NVCC')
CUDA_CFG['cc-mpi'] = (str, 'NVCC')



def load_config(dir=None):
    if dir is None:
        CFG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../config')
    else:
        CFG_DIR = os.path.abspath(dir)


    # parse cuda options
    main_parser = ConfigParser.SafeConfigParser(os.environ)
    main_parser.read(os.path.join(CFG_DIR, 'cuda_default.cfg'))
    for key in CUDA_CFG:
        try:
            t = CUDA_CFG[key][0]
            CUDA_CFG[key] = (t,t(main_parser.get('cuda_ppmd', key)))
        except ConfigParser.InterpolationError:
            pass
        except ConfigParser.NoOptionError:
            pass


    assert CUDA_CFG['cc-main'][1] in config.COMPILERS.keys(), "cc-main CUDA compiler config not found"
    assert CUDA_CFG['cc-mpi'][1] in config.COMPILERS.keys(), "cc-mpi CUDA compiler config not found"


load_config()

LIB_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'lib/'))