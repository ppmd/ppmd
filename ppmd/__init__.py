
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI as _MPI
_is_init = _MPI.Is_initialized()

if _is_init:
    print("Warning MPI was initialised before prefork, this is not supported with OpenMPI.")

from pytools import prefork
prefork.enable_prefork()

__all__ = [ 'mpi',
            'access',
            'cell',
            'data',
            'domain',
            'halo',
            'host',
            'kernel',
            'loop',
            'method',
            'pairloop',
            'pio',
            'runtime',
            'state',
            'opt',
            'utility',
            'coulomb',
            'plain_cell_list']

from . import modules
from . import utility
from . import coulomb
from . import pairloop
from . import loop
from . import data
from . import access
from . import cell
from . import domain
from . import halo
from . import host
from . import kernel
from . import method
from . import mpi
from . import opt
from . import pio
from . import runtime
from . import state
from . import plain_cell_list

abort = mpi.abort
check = mpi.check

mpi.check_pythonhashseed()

