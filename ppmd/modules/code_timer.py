from __future__ import print_function, division

import ctypes

import cgen
import numpy as np

from ppmd.modules.module import Module
from ppmd import mpi

_MPI = mpi.MPI
SUM = _MPI.SUM
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"


class LoopTimer(Module):
    """
    Object to time the runtime of loops. Provides C++ code to be added to the
    generated code.
    """
    def __init__(self):
        self._time = ctypes.c_double(0.0)

    @property
    def time(self):
        """
        Return the current total time.
        """
        return self._time.value

    @property
    def cpu_time(self):
        """
        Return the aggregated cpu time.
        """
        if _MPISIZE == 1:
            return self.time
        else:
            _my_time = np.array(self._time.value)
            _ttime =np.zeros(1)
            _MPIWORLD.Allreduce(_my_time, _ttime, SUM)
            return _ttime[0]

    @property
    def av_time(self):
        """
        Return the aggregated cpu time.
        """
        if _MPISIZE == 1:
            return self.time
        else:
            _my_time = np.array(self._time.value)
            _ttime =np.zeros(1)
            _MPIWORLD.Allreduce(_my_time, _ttime, SUM)
            return _ttime[0]/float(_MPISIZE)

    def get_cpp_headers(self):
        """
        Return the code to include the required header file(s).
        """
        return Code('#include <chrono>\n')

    def get_cpp_headers_ast(self):
        """
        Return the code to include the required header file(s).
        """
        return cgen.Include('chrono')

    def get_cpp_arguments(self):
        """
        Return the code to define arguments to add to the library.
        """
        return Code('double* _loop_timer_return')

    def get_cpp_arguments_ast(self):
        """
        Return the code to define arguments to add to the library.
        """
        return cgen.Pointer(cgen.Value('double', '_loop_timer_return'))


    def get_cpp_pre_loop_code(self):
        """
        Return the code to place before the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t0 ='\
             ' std::chrono::high_resolution_clock::now(); \n'

        return Code(_s)

    def get_cpp_pre_loop_code_ast(self):
        """
        Return the code to place before the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t0 ='\
             ' std::chrono::high_resolution_clock::now(); \n'
        return cgen.Module([cgen.Line(_s)])


    def get_cpp_post_loop_code(self):
        """
        Return the code to place after the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t1 ='\
             ' std::chrono::high_resolution_clock::now(); \n' \
             ' std::chrono::duration<double> _loop_timer_res = _loop_timer_t1'\
             ' - _loop_timer_t0; \n' \
             '*_loop_timer_return += (double) _loop_timer_res.count(); \n'

        return Code(_s)


    def get_cpp_post_loop_code_ast(self):
        """
        Return the code to place after the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t1 ='\
             ' std::chrono::high_resolution_clock::now(); \n' \
             ' std::chrono::duration<double> _loop_timer_res = _loop_timer_t1'\
             ' - _loop_timer_t0; \n' \
             '*_loop_timer_return += (double) _loop_timer_res.count(); \n'
        return cgen.Module([cgen.Line(_s)])


    def get_python_parameters(self):
        """
        Return the parameters to add to the launch of the shared library.
        """
        return ctypes.byref(self._time)