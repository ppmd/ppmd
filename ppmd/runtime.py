__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


# system level
import os

# package level
import config



OPT = config.MAIN_CFG['opt-level'][1]
DEBUG = config.MAIN_CFG['debug-level'][1]
VERBOSE = config.MAIN_CFG['verbose-level'][1]
TIMER = config.MAIN_CFG['timer-level'][1]
BUILD_TIMER = config.MAIN_CFG['build-timer-level'][1]
ERROR_LEVEL = config.MAIN_CFG['error-level'][1]

BUILD_DIR = os.path.abspath(config.MAIN_CFG['build-dir'][1])
LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/'))

MPI_SHARED_MEM = True


try:
    OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
except Exception as e:
    print "error", e
    OMP_NUM_THREADS = 1

print "val", OMP_NUM_THREADS


NUM_THREADS = OMP_NUM_THREADS









