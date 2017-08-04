from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import os
import sys

# package level
import ppmd.config as config

PY_MAJOR_VERSION = sys.version_info[0]

OPT = config.MAIN_CFG['opt-level'][1]
DEBUG = config.MAIN_CFG['debug-level'][1]
VERBOSE = config.MAIN_CFG['verbose-level'][1]
TIMER = config.MAIN_CFG['timer-level'][1]
BUILD_TIMER = config.MAIN_CFG['build-timer-level'][1]
ERROR_LEVEL = config.MAIN_CFG['error-level'][1]

BUILD_DIR = os.path.abspath(config.MAIN_CFG['build-dir'][1])
LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/'))

BUILD_PER_PROC = False

MPI_SHARED_MEM = True

try:
    OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
except Exception as e:
    OMP_NUM_THREADS = None

if OMP_NUM_THREADS is not None:
    NUM_THREADS = OMP_NUM_THREADS
else:
    NUM_THREADS = 1

MPI_DIMS = None


import ppmd.mpi

abort = ppmd.mpi.abort



