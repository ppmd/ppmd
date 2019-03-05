from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import mpi4py
# mpi4py.rc(thread_level='serialized')

from mpi4py import MPI
import sys, atexit, traceback
import ctypes as ct
import numpy as np

import os

if sys.version_info[0] >= 3:
    import queue as Queue
else:
    import Queue


if not MPI.Is_initialized():
    MPI.Init()


# priority queue for module cleanup.
_CLEANUP_QUEUE = Queue.PriorityQueue()


#def _finalise_wrapper():
#    if MPI.Is_initialized():
#        MPI.Finalize()
#_CLEANUP_QUEUE.put((50, _finalise_wrapper))



def _atexit_queue():
    while not _CLEANUP_QUEUE.empty():
        item = _CLEANUP_QUEUE.get()
        item[1]()


atexit.register(_atexit_queue)


mpi_map = {
    ct.c_double: MPI.DOUBLE,
    ct.c_int: MPI.INT,
    int: MPI.INT,
    ct.c_byte: MPI.BYTE
}

# shifts defined as (x, y, z)
recv_modifiers = [
    [-1, -1, -1],  # 0
    [0, -1, -1],  # 1
    [1, -1, -1],  # 2
    [-1, 0, -1],  # 3
    [0, 0, -1],  # 4
    [1, 0, -1],  # 5
    [-1, 1, -1],  # 6
    [0, 1, -1],  # 7
    [1, 1, -1],  # 8

    [-1, -1, 0],  # 9
    [0, -1, 0],  # 10
    [1, -1, 0],  # 11
    [-1, 0, 0],  # 12
    [1, 0, 0],  # 13
    [-1, 1, 0],  # 14
    [0, 1, 0],  # 15
    [1, 1, 0],  # 16

    [-1, -1, 1],  # 17
    [0, -1, 1],  # 18
    [1, -1, 1],  # 19
    [-1, 0, 1],  # 20
    [0, 0, 1],  # 21
    [1, 0, 1],  # 22
    [-1, 1, 1],  # 23
    [0, 1, 1],  # 24
    [1, 1, 1],  # 25
]

tuple_to_direction = {}
for idx, dir in enumerate(recv_modifiers):
    tuple_to_direction[str(dir)] = idx

def enum(**enums):
    return type('Enum', (), enums)

decomposition = enum(spatial=0, particle=1)

# default to spatial decomposition
decomposition_method = decomposition.spatial

Status = MPI.Status


def all_reduce(array):
    rarr = np.zeros_like(array)
    MPI.COMM_WORLD.Allreduce(
        array,
        rarr
    )
    return rarr

###############################################################################
# shared memory mpi handle
###############################################################################

class MPISHM(object):
    """
    This class controls two mpi communicators (assuming MPI3 or higher). 

    The first communicator from:
        MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED,...).
    
    The second a communicator between rank 0 of the shared memory regions.
    """

    def __init__(self, parent_comm=MPI.COMM_WORLD):
        self.init = False
        self.parent_comm = parent_comm
        self.inter_comm = None
        self.intra_comm = None


    def _init_comms(self):
        """
        Initialise the communicators.
        """

        if not self.init:
            assert MPI.VERSION >= 3, "MPI ERROR: mpi4py is not built against"\
                + " a MPI3 or higher MPI distribution."

            self.intra_comm = self.parent_comm.Split_type(MPI.COMM_TYPE_SHARED)

            if self.intra_comm.Get_rank() == 0:
                colour = 0
            else:
                colour = MPI.UNDEFINED

            self.inter_comm = self.parent_comm.Split(color=colour)

            self.init = True

    def _print_comm_info(self):
        self._init_comms()
        print(self.intra_comm.Get_rank(), self.intra_comm.Get_size())
        if self.intra_comm.Get_rank() == 0:
            print(self.inter_comm.Get_rank(), self.inter_comm.Get_size())

    def get_intra_comm(self):
        """
        get communicator for shared memory region.
        """
        self._init_comms()
        return self.intra_comm

    def get_inter_comm(self):
        """
        get communicator between shared memory regions. Only valid on rank 0 of
        the intracomm
        """
        self._init_comms()
        return self.inter_comm

###############################################################################
# shared memory default
###############################################################################

SHMMPI_HANDLE = MPISHM()


###############################################################################
# shared memory mpi handle
###############################################################################

class SHMWIN(object):
    """
    Create a shared memory window in each shared memory region
    """
    def __init__(self, size=None, intracomm=None):
        """
        Allocate a shared memory region.
        :param size: Number of bytes per process.
        :param intracomm: Intracomm to use.
        """
        assert size is not None, "No size passed"
        assert intracomm is not None, "No intracomm passed"
        self._swin = MPI.Win()
        """temp window object."""
        self.win = self._swin.Allocate_shared(size=size, comm=intracomm)
        """Win instance with shared memory allocated"""

        assert self.win.model == MPI.WIN_UNIFIED, "Memory model is not MPI_WIN_UNIFIED"

        self.size = size
        """Size in allocated per process in intercomm"""
        self.intercomm = intracomm
        """Intercomm for RMA shared memory window"""
        self.base = ct.c_void_p(self.win.Get_attr(MPI.WIN_BASE))
        """base pointer for calling rank in shared memory window"""


    def _test(self):
        lib = ct.cdll.LoadLibrary("/home/wrs20/md_workspace/test1.so")

        self.win.Fence()
        MPI.COMM_WORLD.Barrier()

        ptr = ct.c_void_p(self.win.Get_attr(MPI.WIN_BASE))

###############################################################################
# MPI_HANDLE
###############################################################################

def print_str_on_0(comm, *args):
    """
    Method to print on rank 0 to stdout
    """

    if comm.Get_rank() == 0:
        _s = ''
        for ix in args:
            _s += str(ix) + ' '
        print(_s)
        sys.stdout.flush()

    comm.Barrier()

###############################################################################
# cartcomm functions
###############################################################################


def create_cartcomm(comm, dims, periods, reorder_flag):
    """
    Create an mpi cart on the current comm
    """
    COMM = comm.Create_cart(dims, periods, reorder_flag)
    return COMM


def cartcomm_get_move_send_recv_ranks(comm):

    send_ranks = 26*[-1]
    recv_ranks = 26*[-1]

    for ix in range(26):
        direction = recv_modifiers[ix]
        send_ranks[ix] = cartcomm_shift(comm, direction, ignore_periods=True)
        recv_ranks[ix] = cartcomm_shift(comm,
                                        (-1 * direction[0],
                                        -1 * direction[1],
                                        -1 * direction[2]),
                                        ignore_periods=True)
    return send_ranks, recv_ranks


def cartcomm_shift(comm, offset=(0, 0, 0), ignore_periods=False):
    """
    Returns rank of process found at a given offset, will return -1 if no 
    process exists.
    :arg tuple offset: 3-tuple offset from current process.
    """

    if type(offset) is int:
        offset = recv_modifiers[offset]

    _top = cartcomm_top_xyz(comm)
    _per = cartcomm_periods_xyz(comm)
    _dims = cartcomm_dims_xyz(comm)

    _x = _top[0] + offset[0]
    _y = _top[1] + offset[1]
    _z = _top[2] + offset[2]

    _r = [_x % _dims[0], _y % _dims[1], _z % _dims[2]]

    if not ignore_periods:

        if (_r[0] != _x) and _per[0] == 0:
            return -1
        if (_r[1] != _y) and _per[1] == 0:
            return -1
        if (_r[2] != _z) and _per[2] == 0:
            return -1

    return _r[0] + _r[1] * _dims[0] + _r[2] * _dims[0] * _dims[1]


def cartcomm_top_xyz(comm):
    """
    Return the current topology.
    """
    return comm.Get_topo()[2][::-1]

def cartcomm_dims_xyz(comm):
    """
    Return the current dimensions.
    """
    return comm.Get_topo()[0][::-1]


def cartcomm_periods_xyz(comm):
    """
    Return the current periods.
    """
    return comm.Get_topo()[1][::-1]


def check(statement, message):
    if not statement:
        abort(err=message)


def abort(err='-', err_code=0):
    print(80*"=")
    print("MPI:Abort --- COMM_WORLD Rank:", MPI.COMM_WORLD.Get_rank(), '---')
    print(err)
    print(80*"=")
    traceback.print_stack()
    print(80*"=")
    sys.stdout.flush()
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(err_code)

# https://groups.google.com/forum/#!msg/mpi4py/me2TFzHmmsQ/sSF99LE0t9QJ
if MPI.COMM_WORLD.size > 1:
    except_hook = sys.excepthook

    def mpi_excepthook(typ, value, traceback):
        print("ATTEMPTING TO TERMINATE")
        except_hook(typ, value, traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        abort()

    sys.excepthook = mpi_excepthook

_badhashstring = '''

===========================================================================
Environment variable PYTHONHASHSEED is either not set or set to a bad
value. For Bash like shells execute the following or similar before mpirun:

export PYTHONHASHSEED=1234

===========================================================================
'''

def check_pythonhashseed():
    if MPI.COMM_WORLD.size == 1:
        return
    if sys.version_info[0] < 3:
        return
    if 'PYTHONHASHSEED' not in os.environ.keys():
        raise RuntimeError(_badhashstring)
    if int(os.environ['PYTHONHASHSEED'], 10) < 0:
        raise RuntimeError(_badhashstring)
    if int(os.environ['PYTHONHASHSEED'], 10) >= 4294967295:
        raise RuntimeError(_badhashstring)

