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



class GlobalArrayShared(GlobalArrayClassic):
    def __init__(self, size=1, dtype=ctypes.c_double, comm=mpi.MPI.COMM_WORLD, op=mpi.MPI.SUM):
        super().__init__(size=size, dtype=dtype, comm=comm, op=op)
        self.thread_count = 1
        self._threaded = False

    def _init_shared_memory(self):
        if self._threaded is True:
            return
        self._kdata = [np.zeros(shape=self.size, dtype=self.dtype) for tx in range(self.thread_count)]
        self._write_pointers = self._kdata[0].ctypes.get_as_parameter()
        self._read_pointers = self._data.ctypes.get_as_parameter()
        self._threaded = True


