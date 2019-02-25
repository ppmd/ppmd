"""
Optimisation and profiling tools
"""
from __future__ import division, print_function, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


# system level imports
import ctypes
import time
import pickle
import os
import datetime
import glob

# package level imports
from ppmd import mpi, pio

_MPI = mpi.MPI
SUM = _MPI.SUM
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier

def get_timer_accuracy():

    from ppmd.lib import build

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


###############################################################################
# block of code class to be phased out
###############################################################################

class Code(object):
    def __init__(self, init=''):
        self._c = str(init)

    @property
    def string(self):
        return self._c

    def add_line(self, line=''):
        self._c += '\n' + str(line)

    def add(self, code=''):
        self._c += str(code)

    def __iadd__(self, other):
        self.add(code=str(other))
        return self

    def __str__(self):
        return str(self._c)

    def __add__(self, other):
        return Code(self.string + str(other))


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


        t_tmp = self._tt
        self._ts = 0.0
        self._tt = 0.0

        self._running = False
        return t_tmp

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


class SynchronizedTimer(Timer):

    def start(self):
        """
        Start the timer.
        """
        _MPIBARRIER()
        if (self._lo > self._l) and (self._running is False):
            self._ts = time.time()
            self._running = True

    def pause(self):
        """
        Pause the timer.
        """
        _MPIBARRIER()
        if (self._lo > self._l) and (self._running is True):
            self._tt += time.time() - self._ts
            self._ts = 0.0
            self._running = False

    def stop(self, str=''):
        """
        Stop timer and print time.
        :arg string str: string to append after time. If None time printing will be suppressed.
        """
        _MPIBARRIER()
        if (self._lo > self._l) and (self._running is True):
            self._tt += time.time() - self._ts

        if (self._lo > self._l) and str is not None:
            pio.pprint(self._tt, "s :", str)

        self._ts = 0.0

        tt = self._tt
        self._tt = 0.0

        self._running = False
        return tt


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

PROFILE['MPI:rank'] = _MPIRANK


def print_profile(side_by_side=True):
    if not side_by_side:
        for key, value in sorted(PROFILE.items()):
            print(key)
            print('\t', value)
    else:
        m = 0
        for key, value in sorted(PROFILE.items()):
            m = max(m, len(key))
        for key, value in sorted(PROFILE.items()):
            print(key.ljust(m), '  |  ', value)



def dump_profile():

    rank = _MPIRANK
    cwd = os.getcwd()
    prof_dir = os.path.join(cwd, 'PROFILE')
    this_dump = os.path.join(prof_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if rank == 0:
        if not os.path.exists(prof_dir):
            os.mkdir(prof_dir)
        if not os.path.exists(this_dump):
            os.mkdir(this_dump)
    _MPIBARRIER()

    with open(
            os.path.join(this_dump, 'profile_dict.' + str(rank) + '.pkl'), 'wb'
    ) as fh:
        pickle.dump(PROFILE, fh, pickle.HIGHEST_PROTOCOL)


def load_profiles(dir):
    prof_dir = os.path.abspath(dir)
    files = glob.glob(os.path.join(prof_dir, './*'))
    profs = []
    for file in files:
        with open(file, 'r') as fh:
            profs.append(pickle.load(fh))
    return profs


def load_last_profiles():
    prof_dirs = os.path.abspath('./PROFILE/*')
    last_dir = max(glob.iglob(prof_dirs), key=os.path.getctime)

    return load_profiles(last_dir)





