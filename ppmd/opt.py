"""
Optimisation and profiling tools
"""

# system level imports
import ctypes
from mpi4py.MPI import SUM
import numpy as np

# package level imports
import build
import mpi

def get_timer_accuracy():

    t = ctypes.c_double(0.0)

    build.simple_lib_creator(
        '''
        #include <chrono>
        extern "C" void get_chrono_tick(double *t);
        ''',
        '''
        void get_chrono_tick(double *t){

            std::chrono::high_resolution_clock::duration t0(1);
            std::chrono::duration<double> t1 = t0;
            *t = (double) t1.count();
        }
        ''',
        'opt_tick_test')['get_chrono_tick'](ctypes.byref(t))

    return t.value



class LoopTimer(object):
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
        if mpi.MPI_HANDLE.nproc == 1:
            return self.time
        else:
            _my_time = np.array(self._time.value)
            _ttime =np.zeros(1)
            mpi.MPI_HANDLE.comm.Allreduce(_my_time, _ttime, SUM)
            return _ttime[0]

    @property
    def av_time(self):
        """
        Return the aggregated cpu time.
        """
        if mpi.MPI_HANDLE.nproc == 1:
            return self.time
        else:
            _my_time = np.array(self._time.value)
            _ttime =np.zeros(1)
            mpi.MPI_HANDLE.comm.Allreduce(_my_time, _ttime, SUM)
            return _ttime[0]/float(mpi.MPI_HANDLE.nproc)

    def get_cpp_headers(self):
        """
        Return the code to include the required header file(s).
        """
        return build.Code('#include <chrono>\n')

    def get_cpp_arguments(self):
        """
        Return the code to define arguments to add to the library.
        """
        return build.Code('double* _loop_timer_return')

    def get_cpp_pre_loop_code(self):
        """
        Return the code to place before the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t0 ='\
             ' std::chrono::high_resolution_clock::now(); \n'

        return build.Code(_s)

    def get_cpp_post_loop_code(self):
        """
        Return the code to place after the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t1 ='\
             ' std::chrono::high_resolution_clock::now(); \n' \
             ' std::chrono::duration<double> _loop_timer_res = _loop_timer_t1'\
             ' - _loop_timer_t0; \n' \
             '*_loop_timer_return += (double) _loop_timer_res.count(); \n'

        return build.Code(_s)

    def get_python_parameters(self):
        """
        Return the parameters to add to the launch of the shared library.
        """
        return ctypes.byref(self._time)






