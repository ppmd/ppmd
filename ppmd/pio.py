from __future__ import print_function, division
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

##########################################################################
# Parallel IO helper functions.
##########################################################################

# package level
import mpi

# system level
import sys
import datetime
import os

_MPI = mpi.MPI
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier

##########################################################################
# pprint
##########################################################################


def pprint(*args):
    """
    Print a string on stdout using the default MPI handle using rank 0.
    :param string:
    :return:
    """
    mpi.print_str_on_0(_MPIWORLD, *args)


def rprint(*args):
    """
    Print a string on stdout from all procs. preappended with rank id.
    :param string:
    :return:
    """

    _s = ''
    for ix in args:
        _s += str(ix)

    for ix in range(_MPISIZE):
        if _MPIRANK == ix:
            print("rank", _MPIRANK, ":", _s)
            sys.stdout.flush()

        _MPIBARRIER()


class pfprint(object):
    """
    pprint with a copy placed in a file.
    """
    def __init__(self, dirname='./', filename=None):

        if (_MPIRANK == 0) and (not os.path.exists(dirname)):
            os.mkdir(dirname)

        if filename is None:
            _filename = None

            _tf = 'pfprint_' + str(_MPISIZE) + \
                  datetime.datetime.now().strftime("_%H%M%S_%d%m%y")

            if not os.path.exists(os.path.join(dirname, _tf)):
                _filename = _tf
            else:

                _tf += '_'
                for ix in range(100):
                    if not os.path.exists(os.path.join(dirname, _tf + str(ix))):
                        _tf += str(ix)
                        _filename = _tf
                        break

        else:
            _filename = filename

        _filename = dirname + '/' + _filename

        self._fh = None

        if _MPIRANK == 0:
            assert _filename is not None, "No suitable file name found."
            self._fh = open(_filename, 'w')

    def pprint(self, *args):

        if _MPIRANK == 0:

            assert self._fh is not None, "No open file to write to."

            _s = ''
            for ix in args:
                _s += str(ix) + ' '
            print(_s)

            self._fh.write(_s + '\n')

            sys.stdout.flush()

    def pwrite(self, *args):

        if _MPIRANK == 0:

            assert self._fh is not None, "No open file to write to."

            _s = ''
            for ix in args:
                _s += str(ix) + ' '

            self._fh.write(_s + '\n')


    def _get_str(*args):

        _s = ''
        for ix in args:
            _s += str(ix) + ' '
        return _s

    def close(self):
        """
        Close the open file.
        """

        if _MPIRANK == 0:
            self._fh.close()



































