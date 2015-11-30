import os
import time
import pio

######################################################################
# Timer class
######################################################################


class Timer(object):
    """
    Automatic timing class.
    """
    def __init__(self, level_object, level=0, start=False):
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
        if (self._lo.level > self._l) and (self._running is False):
            self._ts = time.time()
            self._running = True

    def pause(self):
        """
        Pause the timer.
        """
        if (self._lo.level > self._l) and (self._running is True):
            self._tt += time.time() - self._ts
            self._ts = 0.0
            self._running = False

    def stop(self, str=''):
        """
        Stop timer and print time.
        :arg string str: string to append after time. If None time printing will be suppressed.
        """
        if (self._lo.level > self._l) and (self._running is True):
            self._tt += time.time() - self._ts

        if self._lo.level > self._l:
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
        if (str is not None) and (self._lo.level > self._l):
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

        if (str is not None) and (self._lo.level > self._l):
            pio.pprint(_tt, "s :", str)

        return _tt

################################################################################################################
# Level class, avoids passing handles everywhere
################################################################################################################

class Level(object):
    """
    Class to hold a level.
    """
    _level = 0

    def __init__(self, level=0):
        self._level = int(level)

    @property
    def level(self):
        """
        Return current debug level.
        """
        return self._level

    @level.setter
    def level(self, level):
        """
        Set a debug level.
        :arg int level: New debug level.
        """
        self._level = int(level)

DEBUG = Level(0)
VERBOSE = Level(0)
TIMER = Level(0)
BUILD_TIMER = Level(0)

################################################################################################################
# Enable class to provide flags to disable/enable major code blocks, eg use cuda y/n
################################################################################################################

class Enable(object):

    def __init__(self, flag=True):
        self._f = flag

    @property
    def flag(self):
        return self._f

    @flag.setter
    def flag(self, val=True):
        self._f = bool(val)

# Toogle this instance of a Enable class to turn off/on gpucuda module.
CUDA_ENABLED = Enable()




##########################################################################################################
# BUILD DIR
##########################################################################################################

class Dir(object):
    """
    Simple container for a string representing a directory.
    :arg str directory: directory.
    """

    def __init__(self, directory):
        self._dir = directory

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, directory):
        self._dir = directory

try:
    _BUILD_DIR = str(os.path.join(os.environ['BUILD_DIR'],''))
except:
    _BUILD_DIR = './build/'


BUILD_DIR = Dir(_BUILD_DIR)
LIB_DIR = Dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/'))






