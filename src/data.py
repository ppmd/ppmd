import ctypes
import particle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import re
import pickle
from mpi4py import MPI
import sys
import math

np.set_printoptions(threshold='nan')

import access

ctypes_map = {ctypes.c_double: 'double', ctypes.c_int: 'int', 'float64': 'double', 'int32': 'int',
              'doublepointerpointer': 'double **', ctypes.c_longlong: 'long long'}
mpi_map = {ctypes.c_double: MPI.DOUBLE, ctypes.c_int: MPI.INT}




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
            return (0, 0, 0)

    @property
    def dims(self):
        """
        Return the current dimensions.
        """
        if self._COMM is not None:
            return self._COMM.Get_topo()[0][::-1]
        else:
            return (1, 1, 1)

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
        if self._COMM is not None:
            self._COMM.Barrier()

    def print_str(self, *args):
        """
        Method to print on rank 0 to stdout
        """

        if self.rank == 0:
            _s = ''
            for ix in args:
                _s += str(ix)
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

    def shift(self, offset=(0, 0, 0)):
        """
        Returns rank of process found at a given offset, will return -1 if no process exists.
        """

        self._check_comm()

        _x = self._top[0] + offset[0]
        _y = self._top[1] + offset[1]
        _z = self._top[2] + offset[2]

        _r = [_x % self._dims[0], _y % self._dims[1], _z % self._dims[2]]

        if (_r[0] != _x) and self._per[0] == 0:
            return -1
        if (_r[1] != _y) and self._per[1] == 0:
            return -1
        if (_r[2] != _z) and self._per[2] == 0:
            return -1

        return _r[0] + _r[1] * self._dims[0] + _r[2] * self._dims[0] * self._dims[1]

###############################################################################################################
# MPI_HANDLE
###############################################################################################################

MPI_HANDLE = MDMPI()

###############################################################################################################
# MPI_HANDLE
###############################################################################################################

def pprint(*args):
    """
    Print a string on stdout using the default MPI handle.
    :param string:
    :return:
    """
    MPI_HANDLE.print_str(*args)

###############################################################################################################
# XYZWrite
###############################################################################################################


def xyz_write(dir_name='./output', file_name='out.xyz', x=None, title='A', sym='A', n_mol=1, rename_override=False):
    """
    Function to write particle positions in a xyz format.
    
    :arg str dirname: Directory to write to default ./output.
    :arg str file_name: Filename to write to default out.xyz.
    :arg Dat X: Particle dat containing positions.
    :arg str title: title of molecule default ABC. 
    :arg str sym: Atomic symbol for particles, default A.
    :arg int N_mol: Number of atoms per molecule default 1.
    :arg bool rename_override: Flagging as True will disable autorenaming of output file.
    """
    assert x is not None, "xyz_write Error: No data."

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(os.path.join(dir_name, file_name)) & (rename_override is not True):
        file_name = re.sub('.xyz', datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.xyz', file_name)
        if os.path.exists(os.path.join(dir_name, file_name)):
            file_name = re.sub('.xyz', datetime.datetime.now().strftime("_%f") + '.xyz', file_name)
            assert os.path.exists(os.path.join(dir_name, file_name)), "XYZWrite Error: No unique name found."

    space = ' '

    f = open(os.path.join(dir_name, file_name), 'w')
    f.write(str(n_mol) + '\n')
    f.write(str(title) + '\n')
    for ix in range(x.npart):
        f.write(str(sym).rjust(3))
        for iy in range(x.ncomp):
            f.write(space + str('%.5f' % x[ix, iy]))
        f.write('\n')
    f.close()


################################################################################################################
# DrawParticles
################################################################################################################


class DrawParticles(object):
    """
    Class to plot n particles with given positions.
    
    :arg int n: Number of particles.
    :arg np.array(n,3) pos: particle positions.
    :arg np.array(3,1) extent:  domain extents.

    """

    def __init__(self, state=None):

        assert state is not None, "DrawParticles error: no state passed."

        self._state = state

        self._Mh = MPI_HANDLE

        self._Dat = None
        self._gids = None
        self._pos = None
        self._gid = None

        self._N = None
        self._NT = None
        self._extents = None

        if (self._Mh.rank == 0) or (self._Mh is None):
            plt.ion()
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._key = ['red', 'blue']
            plt.show(block=False)

    def draw(self):
        """
        Update current plot, use for real time plotting.
        """
        self._N = self._state.n()
        self._NT = self._state.nt()
        self._extents = self._state.domain.extent

        '''Case where all particles are local'''
        if self._Mh is None:
            self._pos = self._state.positions
            self._gid = self._state.global_ids

        else:
            '''Need an mpi handle if not all particles are local'''
            assert self._Mh is not None, "Error: Not all particles are local but mpi_handle = None."

            '''Allocate if needed'''
            if self._Dat is None:
                self._Dat = particle.Dat(self._NT, 3)
            else:
                self._Dat.resize(self._NT)

            if self._gids is None:
                self._gids = ScalarArray(ncomp=self._NT, dtype=ctypes.c_int)
            else:
                self._gids.resize(self._NT)

            _MS = MPI.Status()

            if self._Mh.rank == 0:

                '''Copy the local data.'''
                self._Dat.dat[0:self._N:, ::] = self._state.positions.dat[0:self._N:, ::]
                self._gids[0:self._N:] = self._state.global_ids[0:self._N:]

                _i = self._N  # starting point pos
                _ig = self._N  # starting point gids

                for ix in range(1, self._Mh.nproc):
                    self._Mh.comm.Recv(self._Dat.dat[_i::, ::], ix, ix, _MS)
                    _i += _MS.Get_count(mpi_map[self._Dat.dtype]) / 3

                    self._Mh.comm.Recv(self._gids.dat[_ig::], ix, ix, _MS)
                    _ig += _MS.Get_count(mpi_map[self._gids.dtype])

                self._pos = self._Dat
                self._gid = self._gids
            else:

                self._Mh.comm.Send(self._state.positions.dat[0:self._N:, ::], 0, self._Mh.rank)
                self._Mh.comm.Send(self._state.global_ids.dat[0:self._N:], 0, self._Mh.rank)

        if self._Mh.rank == 0:

            plt.cla()
            plt.ion()
            for ix in range(self._pos.npart):
                self._ax.scatter(self._pos.dat[ix, 0], self._pos.dat[ix, 1], self._pos.dat[ix, 2],
                                 color=self._key[self._gid[ix] % 2])
            self._ax.set_xlim([-0.5 * self._extents[0], 0.5 * self._extents[0]])
            self._ax.set_ylim([-0.5 * self._extents[1], 0.5 * self._extents[1]])
            self._ax.set_zlim([-0.5 * self._extents[2], 0.5 * self._extents[2]])

            self._ax.set_xlabel('x')
            self._ax.set_ylabel('y')
            self._ax.set_zlabel('z')

            plt.draw()
            plt.show(block=False)


################################################################################################################
# Basic Energy Store
################################################################################################################

class BasicEnergyStore(object):
    """
    Depreiciated, use EnergyStore

    Class to contain recorded values of potential energy u, kenetic energy k, total energy q and time T.
    
    :arg int size: Required size of each container.
    """

    def __init__(self, state=None, size=0, mpi_handle=MDMPI()):

        self._state = state

        self._U_store = ScalarArray(initial_value=0.0, ncomp=size, dtype=ctypes.c_double)
        self._K_store = ScalarArray(initial_value=0.0, ncomp=size, dtype=ctypes.c_double)
        self._Q_store = ScalarArray(initial_value=0.0, ncomp=size, dtype=ctypes.c_double)
        self._T_store = ScalarArray(initial_value=0.0, ncomp=size, dtype=ctypes.c_double)

        self._U_c = 0
        self._K_c = 0
        self._Q_c = 0
        self._T_c = 0
        self._T_base = None

        self._Mh = mpi_handle
        self._size = None

    def append_prepare(self, size):

        self._size = size

        if self._T_base is None:
            self._T_base = 0.0
        else:
            self._T_base = self._T_store[-1]

        # Migrate to scalar dats
        self._U_store.concatenate(size)
        self._K_store.concatenate(size)
        self._Q_store.concatenate(size)
        self._T_store.concatenate(size)

    def u_append(self, val):
        """
        Append a value to potential energy.
        
        :arg double val: value to append
        """

        if self._U_c < self._size:
            self._U_store[self._U_c] = val  # float(not(math.isnan(val))) * val
            self._U_c += 1

    def k_append(self, val):
        """
        Append a value to kenetic energy.
        
        :arg double val: value to append
        """

        if self._K_c < self._size:
            self._K_store[self._K_c] = val  # float(not(math.isnan(val))) * val
            self._K_c += 1

    def q_append(self, val):
        """
        Append a value to total energy.
        
        :arg double val: value to append
        """
        if self._Q_c < self._size:
            self._Q_store[self._Q_c] = val  # float(not(math.isnan(val))) * val
            self._Q_c += 1

    def t_append(self, val):
        """
        Append a value to time store.
        
        :arg double val: value to append
        """
        if self._T_c < self._size:
            self._T_store[self._T_c] = val + self._T_base
            self._T_c += 1

    def plot(self):
        """
        Plot recorded energies against time.
        """

        if (self._Mh is not None) and (self._Mh.nproc > 1):

            # data to collect
            _d = [self._Q_store.dat, self._U_store.dat, self._K_store.dat]

            # make a temporary buffer.
            if self._Mh.rank == 0:
                _buff = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _T = self._T_store.dat
                _Q = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _U = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _K = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)

                _Q.dat[::] += self._Q_store.dat[::]
                _U.dat[::] += self._U_store.dat[::]
                _K.dat[::] += self._K_store.dat[::]

                _dl = [_Q.dat, _U.dat, _K.dat]
            else:
                _dl = [None, None, None]

            for _di, _dj in zip(_d, _dl):

                if self._Mh.rank == 0:
                    _MS = MPI.Status()
                    for ix in range(1, self._Mh.nproc):
                        self._Mh.comm.Recv(_buff.dat[::], ix, ix, _MS)
                        _dj[::] += _buff.dat[::]

                else:
                    self._Mh.comm.Send(_di[::], 0, self._Mh.rank)

            if self._Mh.rank == 0:
                _Q = _Q.dat
                _U = _U.dat
                _K = _K.dat

        else:
            _T = self._T_store.dat
            _Q = self._Q_store.dat
            _U = self._U_store.dat
            _K = self._K_store.dat

        if (self._Mh is None) or (self._Mh.rank == 0):
            print "last total", _Q[-1]
            print "last kinetic", _K[-1]
            print "last potential", _U[-1]
            print "============================================="
            print "first total", _Q[0]
            print "first kinetic", _K[0]
            print "first potential", _U[0]

            plt.ion()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)

            ax2.plot(_T, _Q, color='r', linewidth=2)
            ax2.plot(_T, _U, color='g')
            ax2.plot(_T, _K, color='b')

            ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kinetic energy')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Energy')

            fig2.canvas.draw()
            plt.show(block=False)


###############################################################################################################
# Scalar array.
###############################################################################################################
class ScalarArray(object):
    """
    Base class to hold a single floating point property.
    
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    
    """

    def __init__(self, initial_value=None, name=None, ncomp=1, val=None, dtype=ctypes.c_double, max_size=None):
        """
        Creates scalar with given initial value.
        """

        if max_size is None:
            self._max_size = ncomp
        else:
            self._max_size = max_size

        self._dtype = dtype

        if name is not None:
            self._name = name
        self._N1 = ncomp

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._Dat = np.array(initial_value, dtype=self._dtype, order='C')
                self._N1 = initial_value.shape[0]
                self._max_size = self._N1
            elif type(initial_value) == list:
                self._Dat = np.array(np.array(initial_value), dtype=self._dtype, order='C')
                self._N1 = len(initial_value)
                self._max_size = self._N1
            else:
                self._Dat = float(initial_value) * np.ones([self._N1], dtype=self._dtype, order='C')

        elif val is None:
            self._Dat = np.zeros([self._max_size], dtype=self._dtype, order='C')
        elif val is not None:
            self._Dat = np.array([val], dtype=self._dtype, order='C')

        self._A = False
        self._Aarray = None
        self._DatHaloInit = False

    def concatenate(self, size):
        """
        Increase length of scalar array object.
        
        :arg int size: Number of new elements.
        """
        self._Dat = np.concatenate((self._Dat, np.zeros(size, dtype=self._dtype, order='C')))
        self._N1 += size
        if self._A is True:
            self._Aarray = np.concatenate((self._Aarray, np.zeros(size, dtype=self._dtype, order='C')))
            self._Aarray.fill(0.)

    @property
    def dat(self):
        """
        Returns stored data as numpy array.
        """
        return self._Dat

    @dat.setter
    def dat(self, val):
        self._Dat = np.array([val], dtype=self._dtype)

    def __getitem__(self, ix):
        return self._Dat[ix]

    def scale(self, val):
        """
        Scale data array by value val.
        
        :arg double val: Coefficient to scale all elements by.
        """
        # the below seems to cause glibc errors
        self._Dat = self._Dat * np.array([val], dtype=self._dtype)
        '''
        #work around
        if (self._DatHaloInit == False):
            _s = 1
        else:
            _s = 2
        
        
        for ix in range(_s*self._N1):
            self._Dat[ix]=val*self._Dat[ix]
        '''

    def __call__(self, access=access.RW, halo=True):

        return self

    def zero(self):
        """
        Zero all elements in array.
        """
        # causes glibc errors
        self._Dat = np.zeros(self._N1, dtype=self._dtype, order='C')
        '''
        #work around
        for ix in range(self._N1):
            self._Dat[ix]=0.
        '''

    def __setitem__(self, ix, val):
        self._Dat[ix] = np.array([val], dtype=self._dtype)

        if self._A is True:
            self._Aarray[ix] = np.array([val], dtype=self._dtype)
            self._Alength += 1

    def __str__(self):
        return str(self._Dat)

    @property
    def ctypes_data(self):
        """Return ctypes-pointer to data."""
        return self._Dat.ctypes.data_as(ctypes.POINTER(self._dtype))

    @property
    def dtype(self):
        """ Return Dat c data ctype"""
        return self._dtype

    @property
    def ctypes_value(self):
        """Return first value in correct type."""
        return self._dtype(self._Dat[0])

    @property
    def name(self):
        """
        Returns name of particle dat.
        """
        return self._name

    @property
    def ncomp(self):
        """
        Return number of components.
        """
        return self._N1

    @ncomp.setter
    def ncomp(self, val):
        assert val <= self._max_size, "ncomp, max_size error"
        self._N1 = val

    @property
    def min(self):
        """Return minimum"""
        return self._Dat.min()

    @property
    def max(self):
        """Return maximum"""
        return self._Dat.max()

    @property
    def mean(self):
        """Return mean"""
        return self._Dat.mean()

    @property
    def name(self):
        return self._name

    def resize(self, n):
        if n > self._max_size:
            self._max_size = n + (n - self._max_size) * 10
            self._Dat = np.resize(self._Dat, self._max_size)
            # self._N1 = n

    @property
    def end(self):
        """
        Returns end index of array.
        """
        return self._max_size - 1

    @property
    def sum(self):
        """
        Return array sum
        """
        return self._Dat.sum()

    def dat_write(self, dir_name='./output', filename=None, rename_override=False):
        """
        Function to write ScalarArray objects to disk.
        
        :arg str dir_name: directory to write to, default ./output.
        :arg str filename: Filename to write to, default array name or data.SArray if name unset.
        :arg bool rename_override: Flagging as True will disable autorenaming of output file.
        """

        if (self._name is not None) and (filename is None):
            filename = str(self._name) + '.SArray'
        if filename is None:
            filename = 'data.SArray'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if os.path.exists(os.path.join(dir_name, filename)) & (rename_override is not True):
            filename = re.sub('.SArray', datetime.datetime.now().strftime("_%H%M%S_%d%m%y") + '.SArray', filename)
            if os.path.exists(os.path.join(dir_name, filename)):
                filename = re.sub('.SArray', datetime.datetime.now().strftime("_%f") + '.SArray', filename)
                assert os.path.exists(os.path.join(dir_name, filename)), "DatWrite Error: No unquie name found."

        f = open(os.path.join(dir_name, filename), 'w')
        pickle.dump(self._Dat, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def dat_read(self, dir_name='./output', filename=None):
        """
        Function to read Dat objects from disk.
        
        :arg str dir_name: directory to read from, default ./output.
        :arg str filename: filename to read from.
        """

        assert os.path.exists(dir_name), "Read directory not found"
        assert filename is not None, "DatRead Error: No filename given."

        f = open(os.path.join(dir_name, filename), 'r')
        self = pickle.load(f)
        f.close()

    def average_reset(self):
        """Reset and initialises averaging."""
        if self._A is False:

            self._Aarray = np.zeros([self._N1], dtype=self._dtype, order='C')
            self._Alength = 0.0
            self._A = True
        else:
            self._Aarray.fill(0.)
            self._Alength = 0.0

    @property
    def average(self):
        """Returns averages of recorded values since AverageReset was called."""
        # assert self._A == True, "Run AverageReset to initialise or reset averaging"
        if self._A is True:
            return self._Aarray / self._Alength

    def average_stop(self, clean=False):
        """
        Stops averaging values.
        
        :arg bool clean: Flag to free memory allocated to averaging, default False.
        """
        if self._A is True:
            self._A = False
            if clean is True:
                del self._A

    def average_update(self):
        """Copy values from Dat into averaging array"""
        if self._A is True:
            self._Aarray += self._Dat
            self._Alength += 1
        else:
            self.average_reset()
            self._Aarray += self._Dat
            self._Alength += 1

    def init_halo_dat(self):
        """
        Create a secondary dat container.
        """

        if self._DatHaloInit is False:
            self._max_size *= 2
            self._Dat = np.resize(self._Dat, self._max_size)
            self._DatHaloInit = True

    @property
    def dat_halo_init(self):
        """
        Return status of halo dat.
        """
        return self._DatHaloInit


################################################################################################################
# Pointer array.
################################################################################################################

class PointerArray(object):
    """
    Class to store arrays of pointers.
    
    :arg int length: Length of array.
    :arg ctypes.dtype dtype: pointer data type.
    """

    def __init__(self, length, dtype):
        self._length = length
        self._dtype = dtype
        self._Dat = (ctypes.POINTER(self._dtype) * self._length)()

    @property
    def dtype(self):
        """Returns data type."""
        return self._dtype

    @property
    def ctypes_data(self):
        """Returns pointer to start of array."""
        return self._Dat

    @property
    def ncomp(self):
        """Return number of components"""
        return self._length

    def __getitem__(self, ix):
        return self._Dat[ix]

    def __setitem__(self, ix, val):
        self._Dat[ix] = val


################################################################################################################
# Blank arrays.
################################################################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)

################################################################################################################
# Basic Energy Store
################################################################################################################

class EnergyStore(object):
    """
    Class to hold energy data more sensibly

    :arg state state: Input state to track energy of.
    """

    def __init__(self, state=None):

        assert state is not None, "EnergyStore error, no state passed."

        self._state = state
        self._Mh = self._state.mpi_handle

        self._t = []
        self._k = []
        self._u = []
        self._q = []

    def update(self):
        """
        Update energy tracking of tracked state.
        :return:
        """

        if self._state.n() > 0:
            _U_tmp = self._state.u.dat[0]/self._state.nt()
            _U_tmp += 0.5*self._state.u.dat[1]/self._state.nt()

            self._k.append(self._state.k[0]/self._state.nt())
            self._u.append(_U_tmp)
            self._q.append(_U_tmp+(self._state.k[0])/self._state.nt())

        else:
            self._k.append(0.)
            self._u.append(0.)
            self._q.append(0.)

        self._t.append(self._state.time)


    def plot(self):
        """
        Plot the stored energy data.

        :return:
        """

        assert len(self._t) > 0, "EnergyStore error, no data to plot"

        self._T_store = ScalarArray(self._t)
        self._K_store = ScalarArray(self._k)
        self._U_store = ScalarArray(self._u)
        self._Q_store = ScalarArray(self._q)


        if (self._Mh is not None) and (self._Mh.nproc > 1):

            # data to collect
            _d = [self._Q_store.dat, self._U_store.dat, self._K_store.dat]

            # make a temporary buffer.
            if self._Mh.rank == 0:
                _buff = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _T = self._T_store.dat
                _Q = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _U = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)
                _K = ScalarArray(initial_value=0.0, ncomp=self._T_store.ncomp, dtype=ctypes.c_double)

                _Q.dat[::] += self._Q_store.dat[::]
                _U.dat[::] += self._U_store.dat[::]
                _K.dat[::] += self._K_store.dat[::]

                _dl = [_Q.dat, _U.dat, _K.dat]
            else:
                _dl = [None, None, None]

            for _di, _dj in zip(_d, _dl):

                if self._Mh.rank == 0:
                    _MS = MPI.Status()
                    for ix in range(1, self._Mh.nproc):
                        self._Mh.comm.Recv(_buff.dat[::], ix, ix, _MS)
                        _dj[::] += _buff.dat[::]

                else:
                    self._Mh.comm.Send(_di[::], 0, self._Mh.rank)

            if self._Mh.rank == 0:
                _Q = _Q.dat
                _U = _U.dat
                _K = _K.dat

        else:
            _T = self._T_store.dat
            _Q = self._Q_store.dat
            _U = self._U_store.dat
            _K = self._K_store.dat

        if (self._Mh is None) or (self._Mh.rank == 0):
            print "last total", _Q[-1]
            print "last kinetic", _K[-1]
            print "last potential", _U[-1]
            print "============================================="
            print "first total", _Q[0]
            print "first kinetic", _K[0]
            print "first potential", _U[0]

            plt.ion()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)

            ax2.plot(_T, _Q, color='r', linewidth=2)
            ax2.plot(_T, _U, color='g')
            ax2.plot(_T, _K, color='b')

            ax2.set_title('Red: Total energy, Green: Potential energy, Blue: kinetic energy')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Energy')

            fig2.canvas.draw()
            plt.show(block=False)

class PercentagePrinter(object):
    """
    Class to print percentage completion to console.

    :arg float dt: Time step size.
    :arg float t: End time.
    :arg int percent: Percent to print on.
    """
    def __init__(self, dt, t, percent):
        _dt = dt
        _t = t
        self._p = percent
        self._max_it = math.ceil(_t/_dt)
        self._count = 0
        self._curr_p = percent

    def new_times(self, dt, t):
        """
        Change times.

        :arg float dt: Time step size.
        :arg float t: End time.
        :arg int percent: Percent to print on.
        """
        _dt = dt
        _t = t
        self._p = percent
        self._max_it = math.ceil(_t/_dt)
        self._count = 0
        self._curr_p = percent

    def tick(self):
        """
        Method to call per iteration.
        """

        self._count += 1

        if (float(self._count)/self._max_it)*100 > self._curr_p:
            pprint(self._curr_p, "%")
            self._curr_p += self._p











