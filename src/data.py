import ctypes
import particle
import numpy as np

_GRAPHICS = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _GRAPHICS = False

import datetime
import os
import re
import pickle
from mpi4py import MPI
import sys
import math
import build
import runtime
import pio


np.set_printoptions(threshold='nan')

import access

ctypes_map = {ctypes.c_double: 'double', ctypes.c_int: 'int', 'float64': 'double', 'int32': 'int',
              'doublepointerpointer': 'double **', ctypes.c_longlong: 'long long'}
mpi_map = {ctypes.c_double: MPI.DOUBLE, ctypes.c_int: MPI.INT}




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

        self._Mh = runtime.MPI_HANDLE

        self._Dat = None
        self._gids = None
        self._pos = None
        self._gid = None

        self._N = None
        self._NT = None
        self._extents = None

        if (runtime.MPI_HANDLE.rank == 0) and _GRAPHICS:
            plt.ion()
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._key = ['red', 'blue']
            plt.show(block=False)

    def draw(self):
        """
        Update current plot, use for real time plotting.
        """

        if _GRAPHICS:

            self._N = self._state.n()
            self._NT = self._state.nt()
            self._extents = self._state.domain.extent

            '''Case where all particles are local'''
            if self._Mh is None:
                self._pos = self._state.positions
                self._gid = self._state.global_ids

            else:
                '''Need an mpi handle if not all particles are local'''
                assert self._Mh is not None, "Error: Not all particles are local but runtime.MPI_HANDLE = None."

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

    def __init__(self, state=None, size=0):

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

        self._Mh = runtime.MPI_HANDLE
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
        if _GRAPHICS:

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

#####################################################################################
# Scalar array.
#####################################################################################


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
        if type(val) is np.ndarray:
            self._Dat = val
        else:
            self._Dat = np.array([val], dtype=self._dtype)

    def __getitem__(self, ix):
        return self._Dat[ix]

    def scale(self, val):
        """
        Scale data array by value val.
        
        :arg double val: Coefficient to scale all elements by.
        """

        self._Dat = self._Dat * np.array([val], dtype=self._dtype)

    def __call__(self, access=access.RW, halo=True):

        return self

    def zero(self):
        """
        Zero all elements in array.
        """

        self._Dat = np.zeros(self._N1, dtype=self._dtype, order='C')

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

    @property
    def max_size(self):
        """
        Return actual length of array.
        """
        return self._max_size

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


################################################################################################
# Pointer array.
################################################################################################

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


###################################################################################################
# Blank arrays.
###################################################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)

####################################################################################################
# Basic Energy Store
####################################################################################################


class EnergyStore(object):
    """
    Class to hold energy data more sensibly

    :arg state state: Input state to track energy of.
    """

    def __init__(self, state=None):

        assert state is not None, "EnergyStore error, no state passed."

        self._state = state
        self._Mh = runtime.MPI_HANDLE

        self._t = []
        self._k = []
        self._u = []
        self._q = []


    def update(self):
        """
        Update energy tracking of tracked state.
        :return:
        """

        _k = 0.0
        _u = 0.0
        _q = 0.0
        _t = self._state.time

        if self._state.n() > 0:
            _U_tmp = self._state.u.dat[0]/self._state.nt()
            _U_tmp += 0.5*self._state.u.dat[1]/self._state.nt()

            _k = self._state.k[0]/self._state.nt()
            _u = _U_tmp
            _q = _U_tmp+(self._state.k[0])/self._state.nt()

        self._k.append(_k)
        self._u.append(_u)
        self._q.append(_q)
        self._t.append(_t)




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


        '''REPLACE THIS WITH AN MPI4PY REDUCE CALL'''

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

        if (runtime.MPI_HANDLE.rank == 0) and _GRAPHICS:
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

        if runtime.MPI_HANDLE.rank == 0:
            _fh = open('./output/energy.txt', 'w')
            _fh.write("Time Kinetic Potential Total\n")
            for ix in range(len(self._t)):
                _fh.write("%(T)s %(K)s %(P)s %(Q)s\n" % {'T':_T[ix], 'K':_K[ix], 'P':_U[ix], 'Q':_Q[ix]})
            _fh.close()


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
        self.timer = runtime.Timer(runtime.TIMER, 0, start=False)
        self._timing = False

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

        if (self._timing is False) and (runtime.TIMER.level > 0):
            self.timer.start()

        self._count += 1

        if (float(self._count)/self._max_it)*100 > self._curr_p:

            if runtime.TIMER.level > 0:
                pio.pprint(self._curr_p, "%", self.timer.reset(), 's')
            else:
                pio.pprint(self._curr_p, "%")

            self._curr_p += self._p



























