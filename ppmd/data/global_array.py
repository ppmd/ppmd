from __future__ import print_function, division, absolute_import

import ppmd.opt
import ppmd.runtime

"""
This module contains high level arrays and matrices.
"""

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np
import math

# package level
from ppmd import access, mpi, runtime, host, opt

np.set_printoptions(threshold=1000)
SUM = mpi.MPI.SUM


#####################################################################################
# Global Array
#####################################################################################

_global_array_access = (access.R, access.READ, access.INC, access.INC_ZERO, access.INC0)


class GlobalArray(object):
    """
    Class for global data. This class may be: globally set, incremented and
    read. This class is constructed with a MPI reduction operator, currently
    only MPI.SUM is avaialbe, which defines the addition operator. Global
    setting sets all values in the array to the same value across all ranks.
    All calls must be made on all ranks in parent communicator.
    """

    def __new__(self, size=1, dtype=ctypes.c_double, comm=mpi.MPI.COMM_WORLD,
                op=mpi.MPI.SUM, shared_memory=False, ncomp=None):
        if ncomp is not None:
            size = max(size, ncomp)

        assert shared_memory in (False, 'mpi', 'thread', 'omp')
        if shared_memory == 'mpi':
            return GlobalArrayShared(size=size, dtype=dtype, comm=comm, op=op)
        else:
            return GlobalArrayClassic(size=size, dtype=dtype, comm=comm, op=op)

    def __getitem__(self, item):
        pass

    @property
    def ncomp(self):
        return self.size


class GlobalArrayClassic(host._Array):
    """
    Class for global data. This class may be: globally set, incremented and
    read. This class is constructed with a MPI reduction operator, currently
    only MPI.SUM is avaialbe, which defines the addition operator. Global
    setting sets all values in the array to the same value across all ranks.
    All calls must be made on all ranks in parent communicator.
    """

    def __init__(self, size=1, dtype=ctypes.c_double, comm=mpi.MPI.COMM_WORLD, op=mpi.MPI.SUM):
        # if no shared mem, numpy array, if shared mem, view into window
        self._data = None
        # array to swap with self._data to avoid copying in allreduce
        self._rdata = None
        # sync status
        self._sync_status = True

        assert op is mpi.MPI.SUM, "no other reduction operators are currently implemented"

        self.op = op
        """MPI Reduction operation"""
        self.identity_element = 0
        """Identity element for operation"""
        self.size = size
        """Number of elements in the array."""
        self.dtype = dtype
        """Data type of array."""
        self.shared_memory_type = 'thread'

        self.comm = comm

        self._data = np.zeros(shape=size, dtype=dtype)
        self._rdata = np.zeros(shape=size, dtype=dtype)
        self._data2 = np.zeros(shape=size, dtype=dtype)

        self._data[:] = 0

        self._timer = ppmd.opt.Timer(runtime.TIMER)

        self.thread_count = runtime.NUM_THREADS
        self._kdata = None
        self._write_pointers = None
        self._read_pointers = None
        self._threaded = False

    @property
    def ncomp(self):
        return self.size

    def _init_shared_memory(self):
        if self._threaded is True:
            return
        self._kdata = [np.zeros(shape=self.size, dtype=self.dtype) for tx in range(self.thread_count)]

        self._write_pointers = (ctypes.POINTER(self.dtype) * self.thread_count)(
            *[kx.ctypes.data_as(ctypes.POINTER(self.dtype)) for kx in self._kdata])

        self._read_pointers = (ctypes.POINTER(self.dtype) * self.thread_count)(
            *[self._data.ctypes.data_as(ctypes.POINTER(self.dtype)) for kx in range(self.thread_count)])

        self._threaded = True

    def set(self, val):
        self._sync_wait()
        self._data.fill(val)
        self._rdata.fill(self.identity_element)
        if self._threaded:
            self._zero_thread_regions()

        self._sync_status = True

    def __getitem__(self, item):
        self._sync_wait()
        return self._data[item]

    def __call__(self, mode=access.INC):
        assert mode in _global_array_access
        return self, mode

    @property
    def ctype(self):
        return host.ctypes_map[self.dtype]

    def ctypes_data_access(self, mode=access.READ, pair=False, threaded=False):
        self._sync_wait()

        if mode in (access.INC0, access.INC_ZERO):
            self.set(self.identity_element)

        if threaded is False:
            if not mode.write:
                return self.ctypes_data_read
            else:
                return self.ctypes_data_write
        else:
            self._init_shared_memory()
            if not mode.write:
                return self._read_pointers
            else:
                return self._write_pointers

    @property
    def ctypes_data_read(self):
        self._sync_wait()
        return self._data.ctypes.data_as(ctypes.POINTER(self.dtype))

    @property
    def ctypes_data_write(self):
        self._sync_wait()
        return self._rdata.ctypes.data_as(ctypes.POINTER(self.dtype))

    def ctypes_data_post(self, mode=None, threaded=False):

        if mode.write:
            if self._threaded and threaded:
                for kx in self._kdata:
                    self._rdata[:] += kx[:]
                self._zero_thread_regions()
            self._sync_init()

    def _zero_thread_regions(self):
        for kx in self._kdata:
            kx.fill(self.identity_element)

    def _sync_init(self):
        self._timer.start()
        self._sync_status = False

        self.comm.Allreduce(self._rdata, self._data2, self.op)
        self._data[:] += self._data2[:]
        self._rdata.fill(self.identity_element)

        self._timer.pause()

        opt.PROFILE[
            self.__class__.__name__ + ':{}--{}:{}:'.format(self.dtype, self.size, id(self))
        ] = (self._timer.time())

    def _sync_wait(self):
        if self._sync_status:
            return
        # nothing to do until iallreduce is implemented
        self._sync_status = True


class GlobalArrayShared(host._Array):
    """
    Class for global data. This class may be: globally set, incremented and
    read. This class is constructed with a MPI reduction operator, currently
    only MPI.SUM is avaialbe, which defines the addition operator. Global
    setting sets all values in the array to the same value across all ranks.
    All calls must be made on all ranks in parent communicator.
    """

    def __init__(self, size=1, dtype=ctypes.c_double, comm=mpi.MPI.COMM_WORLD,
                 op=mpi.MPI.SUM):
        # if no shared mem, numpy array, if shared mem, view into window
        self._data = None
        # array to swap with self._data to avoid copying in allreduce
        self._rdata = None
        # MPI shared RMA window
        self._win = None
        # sync status
        self._sync_status = True

        assert op is mpi.MPI.SUM, "no other reduction operators are currently implemented"

        self._timer = ppmd.opt.Timer(runtime.TIMER)

        self.op = op
        """MPI Reduction operation"""
        self.identity_element = 0
        """Identity element for operation"""
        self.size = size
        """Number of elements in the array."""
        self.dtype = dtype
        """Data type of array."""

        self.shared_memory_type = 'mpi'

        self.shared_memory = (mpi.MPI.VERSION >= 3) and runtime.MPI_SHARED_MEM
        """True if shared memory is enabled"""
        self.comm = comm

    @property
    def comm(self):
        return self._comm

    @comm.setter
    def comm(self, comm):
        self._comm = comm
        # aquire/create split communicator for window
        if comm is mpi.MPI.COMM_WORLD and self.shared_memory:
            self._split_comm = mpi.SHMMPI_HANDLE
            self._redcomm = self._split_comm.get_inter_comm()
        elif self.shared_memory:
            self._split_comm = mpi.MPISHM(parent_comm=comm)
            self._redcomm = self._split_comm.get_inter_comm()
        else:
            # problems
            print("critical error in GlobalArrayShared init")

        # create shared memory windows and numpy views

        # the write space
        self._win = mpi.SHMWIN(size=self.size * ctypes.sizeof(self.dtype),
                               intracomm=mpi.SHMMPI_HANDLE.get_intra_comm())

        if hasattr(self._win.win, 'memory'):
            self._data_memview = np.array(self._win.win.memory, copy=False)
            self._data = self._data_memview.view(dtype=self.dtype)
        elif hasattr(self._win.win, 'tomemory'):
            self._data_memview = np.frombuffer(self._win.win.tomemory(),
                                               dtype=self.dtype)
            self._data = self._data_memview.view()
        else:
            raise RuntimeError('cannot get shared memory from window')

        # reduction and read space
        rsize = int(self._split_comm.get_intra_comm().Get_rank() < 1) * \
            self.size * ctypes.sizeof(self.dtype)

        # window 1 ===========================================================
        self._rwin = mpi.SHMWIN(size=rsize,
                                intracomm=mpi.SHMMPI_HANDLE.get_intra_comm())
        # view into memory on node rank 0
        rwin_root_memview = self._rwin.win.Shared_query(0)[0]
        self._rdata_memview = np.array(rwin_root_memview, copy=False)
        self._rdata = self._rdata_memview.view(dtype=self.dtype)

        # window 2 ===========================================================
        self._rwin2 = mpi.SHMWIN(size=rsize,
                                 intracomm=mpi.SHMMPI_HANDLE.get_intra_comm())
        # view into memory on node rank 0
        rwin_root_memview2 = self._rwin2.win.Shared_query(0)[0]
        self._rdata_memview2 = np.array(rwin_root_memview2, copy=False)
        self._rdata2 = self._rdata_memview2.view(dtype=self.dtype)

        self._flip = True
        # ====================================================================

        # reduction sizes
        self._msize = None
        self._lsize = self._split_comm.get_intra_comm().Get_size()
        self._lrank = self._split_comm.get_intra_comm().Get_rank()
        if self.size <= self._lsize:
            self._msize = int(self._lrank < self.size)
            self._mstart = self._lrank
        else:
            bsize = int(math.floor(float(self.size) / self._lsize))
            self._msize = bsize
            mrem = (self.size - self._msize * self._lsize) % self._lsize
            self._msize += int(self._lrank < mrem)
            self._mstart = int(self._lrank < mrem) * self._lrank * self._msize + \
                int(self._lrank >= mrem) * (mrem * (bsize + 1) + (self._lrank - mrem) * bsize)

        # view into all of shared memory, list of arrays
        rwin_root_memview = [self._win.win.Shared_query(ix)[0] for ix in range(self._lsize)]
        self._data_root_memview = [np.array(rwin_root_memview[ix], copy=False) for ix in range(self._lsize)]
        self._data_root = [self._data_root_memview[ix].view(dtype=self.dtype) for ix in range(self._lsize)]

    @property
    def ncomp(self):
        return self.size

    def set(self, val):
        self._sync_wait()

        self._split_comm.get_intra_comm().Barrier()
        self._rwin.win.Fence()
        self._rdata[self._mstart:self._mstart + self._msize:] = val
        self._rwin.win.Fence()
        self._split_comm.get_intra_comm().Barrier()

        self._data.fill(self.identity_element)
        self._sync_status = True

    def __getitem__(self, item):
        self._sync_wait()
        return np.array(self._rdata[item])

    def __call__(self, mode=access.INC):
        assert mode in _global_array_access
        return self, mode

    @property
    def ctype(self):
        return host.ctypes_map[self.dtype]

    def ctypes_data_access(self, mode=None, pair=False, threaded=False):
        assert threaded is False, "this global array is not thread safe"

        self._sync_wait()

        assert mode in _global_array_access

        if mode in (access.INC0, access.INC_ZERO):
            self.set(0)

        if mode.write:
            return self._win.base
        else:
            return self._rdata.ctypes.data_as(ctypes.POINTER(self.dtype))

    def ctypes_data_post(self, mode=None, threaded=False):
        assert threaded is False, "this global array is not thread safe"
        if mode.write:
            self._sync_init()

    def _sync_init(self):
        self._timer.start()

        self._sync_status = False

        self._split_comm.get_intra_comm().Barrier()
        self._rwin.win.Fence()

        for rx in range(self._lsize):
            self._rdata[self._mstart:self._mstart + self._msize:] += \
                self._data_root[rx][self._mstart:self._mstart + self._msize:]

        self._rwin.win.Fence()
        self._split_comm.get_intra_comm().Barrier()

        if self._lrank == 0:
            self._redcomm.Allreduce(self._rdata, self._rdata2, self.op)

        self._data.fill(self.identity_element)

        self._split_comm.get_intra_comm().Barrier()

        if self._flip:
            self._rdata = self._rdata_memview2.view(dtype=self.dtype)
            self._rdata2 = self._rdata_memview.view(dtype=self.dtype)
        else:
            self._rdata = self._rdata_memview.view(dtype=self.dtype)
            self._rdata2 = self._rdata_memview2.view(dtype=self.dtype)
        self._flip = not self._flip

        self._split_comm.get_intra_comm().Barrier()

        self._timer.pause()

        opt.PROFILE[
            self.__class__.__name__ + ':{}--{}:{}:'.format(self.dtype, self.size, id(self))
        ] = (self._timer.time())

    def _sync_wait(self):
        if self._sync_status:
            return
        self._sync_status = True
        self._split_comm.get_intra_comm().Barrier()

