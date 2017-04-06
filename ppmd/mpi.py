__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
from mpi4py import MPI
import sys
import ctypes as ct
import numpy as np

#package level
import runtime
import pio
import os


mpi_map = {ct.c_double: MPI.DOUBLE, ct.c_int: MPI.INT, int: MPI.INT}

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



###############################################################################################################
# MDMPI
###############################################################################################################

class MDMPI(object):
    """
    Class to store a MPI communicator such that it can be used everywhere (bottom level of hierarchy).
    """
    def __init__(self):
        self._COMM = MPI.COMM_WORLD
        self._p = (0, 0, 0)

    @property
    def comm(self):
        """
        Return the current communicator.
        """
        return self._COMM

    def create_cart(self, dims, periods, reorder_flag):
        """
        Create an mpi cart on the current comm
        """
        self._COMM = MPI.COMM_WORLD.Create_cart(dims, periods, reorder_flag)

        if runtime.VERBOSE > 1:
            pio.pprint("Processor count ", self.nproc, " Processor layout ", self.dims)


    @property
    def fortran_comm(self):
        return self._COMM.py2f()

    @comm.setter
    def comm(self, new_comm=None):
        """
        Set the current communicator.
        """
        assert new_comm is not None, "MDMPI error: no new communicator assigned."
        self._COMM = new_comm

    def __call__(self):
        """
        Return the current communicator.
        """
        return self._COMM

    @property
    def rank(self):
        """
        Return the current rank.
        """
        if self._COMM is not None:
            return self._COMM.Get_rank()
        else:
            return 0

    @property
    def nproc(self):
        """
        Return the current size.
        """
        if self._COMM is not None:
            return self._COMM.Get_size()
        else:
            return 1

    @property
    def top(self):
        """
        Return the current topology.
        """
        if self._COMM is not None:
            return self._COMM.Get_topo()[2][::-1]
        else:
            return 0, 0, 0

    @property
    def dims(self):
        """
        Return the current dimensions.
        """
        if self._COMM is not None:
            return self._COMM.Get_topo()[0][::-1]
        else:
            return 1, 1, 1

    @property
    def periods(self):
        """
        Return the current periods.
        """
        if self._COMM is not None:
            return self._COMM.Get_topo()[1][::-1]
        else:
            return self._p

    def set_periods(self, p=None):
        """
        set periods (if for some reason mpi4py does not set these this prives a soln.
        """
        assert p is not None, "Error no periods passed"
        self._p = p

    def barrier(self):
        """
        alias to comm barrier method.
        """

        # MPI.COMM_WORLD.Barrier()
        if self._COMM is not None:
            self._COMM.Barrier()


    def print_str(self, *args):
        """
        Method to print on rank 0 to stdout
        """

        if self.rank == 0:
            _s = ''
            for ix in args:
                _s += str(ix) + ' '
            print _s
            sys.stdout.flush()

        self.barrier()


    def _check_comm(self):
        self._top = self._COMM.Get_topo()[2][::-1]
        self._per = self._COMM.Get_topo()[1][::-1]
        self._dims = self._COMM.Get_topo()[0][::-1]

    @property
    def query_boundary_exist(self):
        """
        Return for each direction:
        Flag if process is a boundary edge or interior edge 1 or 0.

        Xl 0, Xu 1
        Yl 2, Yu 3
        Zl 4, Zu 5
        """

        self._check_comm()

        _sf = range(6)
        for ix in range(3):
            if self._top[ix] == 0:
                _sf[2 * ix] = 1
            else:
                _sf[2 * ix] = 0
            if self._top[ix] == self._dims[ix] - 1:
                _sf[2 * ix + 1] = 1
            else:
                _sf[2 * ix + 1] = 0
        return _sf

    @property
    def query_halo_exist(self):
        """
        Return for each direction:
        Flag if process has a halo on each face.

        Xl 0, Xu 1
        Yl 2, Yu 3
        Zl 4, Zu 5

        """

        self._check_comm()

        _sf = range(6)
        for ix in range(3):
            if self._top[ix] == 0:
                _sf[2 * ix] = self._per[ix]
            else:
                _sf[2 * ix] = 1
            if self._top[ix] == self._dims[ix] - 1:
                _sf[2 * ix + 1] = self._per[ix]
            else:
                _sf[2 * ix + 1] = 1
        return _sf

    def shift(self, offset=(0, 0, 0), ignore_periods=False):
        """
        Returns rank of process found at a given offset, will return -1 if no process exists.

        :arg tuple offset: 3-tuple offset from current process.
        """

        if type(offset) is int:
            offset = recv_modifiers[offset]

        self._check_comm()

        _x = self._top[0] + offset[0]
        _y = self._top[1] + offset[1]
        _z = self._top[2] + offset[2]

        _r = [_x % self._dims[0], _y % self._dims[1], _z % self._dims[2]]

        if not ignore_periods:

            if (_r[0] != _x) and self._per[0] == 0:
                return -1
            if (_r[1] != _y) and self._per[1] == 0:
                return -1
            if (_r[2] != _z) and self._per[2] == 0:
                return -1

        return _r[0] + _r[1] * self._dims[0] + _r[2] * self._dims[0] * self._dims[1]

    def get_move_send_recv_ranks(self):
        send_ranks = range(26)
        recv_ranks = range(26)

        for ix in range(26):
            direction = recv_modifiers[ix]
            send_ranks[ix] = self.shift(direction, ignore_periods=True)
            recv_ranks[ix] = self.shift((-1 * direction[0],
                                         -1 * direction[1],
                                         -1 * direction[2]),
                                         ignore_periods=True)

        return send_ranks, recv_ranks




###############################################################################
# MPI_HANDLE
###############################################################################

# Main MPI communicatior used by program.

MPI_HANDLE = None
def reset_mpi():
    global MPI_HANDLE
    MPI_HANDLE = MDMPI()

reset_mpi()


Status = MPI.Status


def all_reduce(array):
    rarr = np.zeros_like(array)
    MPI_HANDLE.comm.Allreduce(
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

    def __init__(self):
        self.init = False
        self.inter_comm = None
        self.intra_comm = None

    def _init_comms(self):
        """
        Initialise the communicators.
        """

        if not self.init:
            assert MPI.VERSION >= 3, "MPI ERROR: mpi4py is not built against"\
                + " a MPI3 or higher MPI distribution."

            self.intra_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

            if self.intra_comm.Get_rank() == 0:
                colour = 0
            else:
                colour = MPI.UNDEFINED

            self.inter_comm = MPI.COMM_WORLD.Split(color=colour)

            self.init = True

    def _print_comm_info(self):
        self._init_comms()
        print self.intra_comm.Get_rank(), self.intra_comm.Get_size()
        if self.intra_comm.Get_rank() == 0:
            print self.inter_comm.Get_rank(), self.inter_comm.Get_size()

    def get_intra_comm(self):
        """
        get communicator for shared memory region.
        """
        self._init_comms()
        return self.intra_comm

    def get_inter_comm(self):
        """
        get communicator between shared memory regions.
        """
        self._init_comms()
        if self.intra_comm.Get_rank() != 0:
            print "warning this MPI comm is undefined on this rank"
        return self.inter_comm

###############################################################################
# sared memory default
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

        print "\t", rank, ptr

        self.win.Fence()
        MPI.COMM_WORLD.Barrier()

        lib['test1'](
            ptr,
            ct.c_int(size),
            ct.c_int(rank)
        )

        self.win.Fence()
        MPI.COMM_WORLD.Barrier()
        lib['test2'](
            ptr,
            ct.c_int(size),
            ct.c_int(rank)
        )

        self.win.Fence()
        MPI.COMM_WORLD.Barrier()
        lib['test3'](
            ptr,
            ct.c_int(size),
            ct.c_int(rank)
        )

        self.win.Fence()
        MPI.COMM_WORLD.Barrier()

        lib['test1'](
            ptr,
            ct.c_int(size),
            ct.c_int(rank)
        )







