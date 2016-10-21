"""
Optimisation and profiling tools
"""

# system level imports
import ctypes
from mpi4py.MPI import SUM
import numpy as np
import time
import cgen

# package level imports
import build
import mpi
import pio
import host
import runtime

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

######################################################################
# Timer class
######################################################################



class Timer(object):
    """
    Automatic timing class.
    """
    def __init__(self, level_object=1, level=0, start=False):
        self._lo = level_object
        self._l = level
        self._ts = 0.0
        self._tt = 0.0
        self._running = False

        if start:
            self.start()

    def start(self):
        """
        Start the timer.
        """
        if (self._lo > self._l) and (self._running is False):
            self._ts = time.time()
            self._running = True

    def pause(self):
        """
        Pause the timer.
        """
        if (self._lo > self._l) and (self._running is True):
            self._tt += time.time() - self._ts
            self._ts = 0.0
            self._running = False

    def stop(self, str=''):
        """
        Stop timer and print time.
        :arg string str: string to append after time. If None time printing will be suppressed.
        """
        if (self._lo > self._l) and (self._running is True):
            self._tt += time.time() - self._ts

        if (self._lo > self._l) and str is not None:
            pio.pprint(self._tt, "s :", str)

        self._ts = 0.0
        self._tt = 0.0

        self._running = False

    def time(self, str=None):
        """
        Return current total time.
        :arg string str: string to append after time. If None time printing will be suppressed.
        :return: Current total time as float.
        """
        if (str is not None) and (self._lo > self._l):
            pio.pprint(self._tt, "s :", str)

        return self._tt

    def reset(self, str=None):
        """
        Resets the timer. Returns the time taken up until the reset.
        :arg string str: If not None will print time followed by string.
        """
        _tt = self._tt + time.time() - self._ts
        self._tt = 0.0
        self._ts = time.time()

        if (str is not None) and (self._lo > self._l):
            pio.pprint(_tt, "s :", str)

        return _tt






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

    def get_cpp_headers_ast(self):
        """
        Return the code to include the required header file(s).
        """
        return cgen.Include('chrono')

    def get_cpp_arguments(self):
        """
        Return the code to define arguments to add to the library.
        """
        return build.Code('double* _loop_timer_return')

    def get_cpp_arguments_ast(self):
        """
        Return the code to define arguments to add to the library.
        """
        return cgen.Pointer(cgen.Value(host.double_str, '_loop_timer_return'))


    def get_cpp_pre_loop_code(self):
        """
        Return the code to place before the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t0 ='\
             ' std::chrono::high_resolution_clock::now(); \n'

        return build.Code(_s)

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

        return build.Code(_s)


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





class SynchronizedTimer(Timer):

    def start(self):
        """
        Start the timer.
        """
        mpi.MPI_HANDLE.barrier()
        if (self._lo > self._l) and (self._running is False):
            self._ts = time.time()
            self._running = True

    def pause(self):
        """
        Pause the timer.
        """
        mpi.MPI_HANDLE.barrier()
        if (self._lo > self._l) and (self._running is True):
            self._tt += time.time() - self._ts
            self._ts = 0.0
            self._running = False

    def stop(self, str=''):
        """
        Stop timer and print time.
        :arg string str: string to append after time. If None time printing will be suppressed.
        """
        mpi.MPI_HANDLE.barrier()
        if (self._lo > self._l) and (self._running is True):
            self._tt += time.time() - self._ts

        if (self._lo > self._l) and str is not None:
            pio.pprint(self._tt, "s :", str)

        self._ts = 0.0
        self._tt = 0.0

        self._running = False


PROFILE = {}
"""
Dict available module wide for profiling. Recommended format along lines of:

{
    'description'
:
    (
        total_time_taken
    )
}
"""


def print_profile():
    for key, value in sorted(PROFILE.items()):
        print key
        print '\t',value










