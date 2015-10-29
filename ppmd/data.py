"""
This module contains high level arrays and matrices.
"""

import host
import ctypes
import numpy as np
import access
import halo
import mpi
import cell
import kernel
import build
import runtime
import gpucuda


np.set_printoptions(threshold=1000)


"""
rst_doc{

.. contents:

data Module
===========

.. automodule:: data

Scalar Array
~~~~~~~~~~~~

The class :class:`~data.ScalarArray` is a generic one dimensional array that should be used
to store data within simulations that is not associated with any particular particles. For
example the kinetic energy of the system or the array used to bin values when calculating a
radial distribution.

.. autoclass:: data.ScalarArray
    :show-inheritance:
    :undoc-members:
    :members:


Particle Dat
~~~~~~~~~~~~

This classes should be considered as a two dimensional matrix with each row storing the properties
of a particle. The order of rows in relation to which particle they correspond to should always
be conserved. This is the default behaviour of any sorting methods implemented in this framework.

.. autoclass:: data.ParticleDat
    :show-inheritance:
    :undoc-members:
    :members:

Typed Dat
~~~~~~~~~

Instances of this class should be used to store properties of particles which are common to
multiple particles e.g. mass.


.. autoclass:: data.TypedDat
    :show-inheritance:
    :undoc-members:
    :members:


}rst_doc
"""


#####################################################################################
# Scalar array.
#####################################################################################


class ScalarArray(host.Array):
    """
    Class to hold an array of scalar values.
    
    :arg initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    :arg int ncomp: Number of components.
    :arg dtype: Data type. Should be a ctypes C data type.
    """

    def __init__(self, initial_value=None, name=None, ncomp=1, dtype=ctypes.c_double):
        """
        Creates scalar with given initial value.
        """

        self.halo_aware = False
        """How to handle writes to this dat in a reduction sense. """ \
        """In general for a reduction in a pair loop a write will occur once per pair """ \
        """In the case where one of the pair is in a halo, the write will occur if the ith""" \
        """particle is not in the halo?"""

        self.name = name
        """Name of ScalarArray instance."""

        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)

        self._A = False
        self._Aarray = None
        self._cuda_dat = None

    def __call__(self, mode=access.RW, halo=True):
        return self, mode

    def __setitem__(self, ix, val):
        self.dat[ix] = np.array([val], dtype=self.dtype)

        if self._A is True:
            self._Aarray[ix] = np.array([val], dtype=self.dtype)
            self._Alength += 1

    def __str__(self):
        return str(self.dat)

    def __repr__(self):
        return str(self.dat)

    def resize(self, new_length):
        """
        Increase the size of the array.
        :param int new_length: New array length.
        """
        if new_length > self.ncomp:
            self.realloc(new_length)

    def scale(self, val):
        """
        Scale data array by given value.

        :arg val: Coefficient to scale all elements by.
        """
        self.dat = self.dat * np.array([val], dtype=self.dtype)

    @property
    def ctypes_value(self):
        """:return: first value in correct type."""
        return self.dtype(self.dat[0])

    @property
    def min(self):
        """:return: The minimum in the array."""
        return self.dat.min()

    @property
    def max(self):
        """:return: The maximum value in the array."""
        return self.dat.max()

    @property
    def mean(self):
        """:return: The mean value in the array."""
        return self.dat.mean()

    @property
    def sum(self):
        """
        :return: The array sum.
        """
        return self.dat.sum()

    @property
    def average(self):
        """:return: averages of recorded values since AverageReset was called."""
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
            self._Aarray += self.dat
            self._Alength += 1
        else:
            self.average_reset()
            self._Aarray += self.dat
            self._Alength += 1




###################################################################################################
# Blank arrays.
###################################################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
"""Empty integer :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""


NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)
"""Empty double :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""

###################################################################################################
# ParticleDat.
###################################################################################################


class ParticleDat(host.Matrix):
    """
    Base class to hold properties of particles. This could be considered as a two dimensional matrix
    with each row representing the stored properties of a particle.

    :arg int npart: Number of particles (Number of row in matrix).
    :arg int ncol: Dimension of property to store per particle (Number of columns in matrix).
    :arg initial_value: Value to initialise array with, default zeros.
    :arg str name: Collective name of stored vars eg positions.
    """

    def __init__(self, npart=1, ncomp=1, initial_value=None, name=None, dtype=ctypes.c_double, max_npart=None):
        # version ids. Internal then halo.
        self._vid_int = 0
        self._vid_halo = -1

        self.name = name
        """:return: The name of the ParticleDat instance."""

        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)

            self.max_npart = self.nrow
            """:return: The maximum number of particles which can be stored within this particle dat."""

            self.npart = self.nrow
            """:return: The number of particles with properties stored in the particle dat."""

            self.ncomp = self.ncol
            """:return: The number of components stored for each particle."""

        else:
            if max_npart is None:
                max_npart = npart
            self._create_zeros(max_npart, ncomp, dtype)
            self.max_npart = self.nrow
            self.npart = self.nrow
            self.ncomp = self.ncol

        self._halo_packing_lib = None
        self._halo_packing_buffer = None
        self._cell_contents_recv = None

        self.halo_start = self.npart
        """:return: The starting index of the halo region of the particle dat. """

        self.npart_halo = 0
        """:return: The number of particles currently stored within the halo region of the particle dat."""

        self._cuda_dat = None

    @property
    def dat(self):
        self._vid_int += 1
        return self._dat

    @dat.setter
    def dat(self, value):
        self._vid_int += 1
        self._dat = value

    def set_val(self, val):
        """
        Set all the entries in the particle dat to the same specified value.

        :param val: Value to set all entries to.
        """
        self.dat[..., ...] = val

    @property
    def npart_total(self):#
        """
        Get the total number of particles in the dat including halo particles.

        :return:
        """
        return self.npart + self.npart_halo


    def __getitem__(self, ix):
        return self.dat[ix]

    def __setitem__(self, ix, val):
        self._vid_int += 1
        self.dat[ix] = val

    def __str__(self):
        return str(self.dat[::])

    def __repr__(self):
        self.__str__()

    def __call__(self, mode=access.RW, halo=True):

        return self, mode

    def ctypes_data_access(self, mode=access.RW):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """
        if mode.read:
            if (self._vid_int > self._vid_halo) and cell.cell_list.halos_exist is True:
                self.halo_exchange()

                self._vid_halo = self._vid_int

        return self.dat.ctypes.data_as(ctypes.POINTER(self.dtype))

    def ctypes_data_post(self, mode=access.RW):
        """
        Call after excuting a method on the data.
        :arg access mode: Access type required by the calling method.
        """
        if mode.write:
            self._vid_int += 1

    def halo_start_shift(self, shift):
        """
        Shift the starting point of the halo in the particle dat by the specified shift.
        :param int shift: Offset to shift by.
        """

        self.halo_start += shift
        self.npart_halo = self.halo_start - self.npart

    def halo_start_set(self, index):
        """
        Set the start of the halo region in the particle dat to the specified index.
        :param int index: Index to set to.
        """
        if index < self.npart:
            if index >= 0:
                self.npart = index
                self.halo_start = index

        else:
            self.halo_start_reset()

        self.npart_halo = 0

    def halo_start_reset(self):
        """
        Reset the starting postion of the halo region in the particle dat to the end of the
        local particles.
        :return:
        """
        self.halo_start = self.npart
        self.npart_halo = 0

    def resize(self, n):
        """
        Resize particle dat to be at least a certain size, does not resize if already large enough.
        :arg int n: New minimum size.
        """

        if n > self.max_npart:
            self.max_npart = n
            self.realloc(n, self.ncol)

    def halo_exchange(self):
        """
        Perform a halo exchange for the particle dat. WIP currently only functional for positions.
        """

        self.halo_pack()
        self._transfer_unpack()


    def _setup_halo_packing(self):
        """
        Setup the halo packing shared library and buffer space
        """
        if self.name == 'positions':
            _shift_code = '+ CSA[LINIDX_2D(%(NCOMP)s,ix,cx)]' % {'NCOMP': self.ncomp}
            _args = {'CSA': host.NullDoubleArray}

            '''Calculate flag to determine if a boundary between processes is also a boundary in domain.'''
            _bc_flag = [0, 0, 0, 0, 0, 0]
            for ix in range(3):
                if mpi.MPI_HANDLE.top[ix] == 0:
                    _bc_flag[2 * ix] = 1
                if mpi.MPI_HANDLE.top[ix] == mpi.MPI_HANDLE.dims[ix] - 1:
                    _bc_flag[2 * ix + 1] = 1

            _extent = cell.cell_list.domain.extent

            '''Shifts to apply to positions when exchanging over boundaries.'''
            _cell_shifts = [
                [-1 * _extent[0] * _bc_flag[1], -1 * _extent[1] * _bc_flag[3],
                 -1 * _extent[2] * _bc_flag[5]],
                [0., -1 * _extent[1] * _bc_flag[3], -1 * _extent[2] * _bc_flag[5]],
                [_extent[0] * _bc_flag[0], -1 * _extent[1] * _bc_flag[3], -1 * _extent[2] * _bc_flag[5]],
                [-1 * _extent[0] * _bc_flag[1], 0., -1 * _extent[2] * _bc_flag[5]],
                [0., 0., -1 * _extent[2] * _bc_flag[5]],
                [_extent[0] * _bc_flag[0], 0., -1 * _extent[2] * _bc_flag[5]],
                [-1 * _extent[0] * _bc_flag[1], _extent[1] * _bc_flag[2], -1 * _extent[2] * _bc_flag[5]],
                [0., _extent[1] * _bc_flag[2], -1 * _extent[2] * _bc_flag[5]],
                [_extent[0] * _bc_flag[0], _extent[1] * _bc_flag[2], -1 * _extent[2] * _bc_flag[5]],

                [-1 * _extent[0] * _bc_flag[1], -1 * _extent[1] * _bc_flag[3], 0.],
                [0., -1 * _extent[1] * _bc_flag[3], 0.],
                [_extent[0] * _bc_flag[0], -1 * _extent[1] * _bc_flag[3], 0.],
                [-1 * _extent[0] * _bc_flag[1], 0., 0.],
                [_extent[0] * _bc_flag[0], 0., 0.],
                [-1 * _extent[0] * _bc_flag[1], _extent[1] * _bc_flag[2], 0.],
                [0., _extent[1] * _bc_flag[2], 0.],
                [_extent[0] * _bc_flag[0], _extent[1] * _bc_flag[2], 0.],

                [-1 * _extent[0] * _bc_flag[1], -1 * _extent[1] * _bc_flag[3], _extent[2] * _bc_flag[4]],
                [0., -1 * _extent[1] * _bc_flag[3], _extent[2] * _bc_flag[4]],
                [_extent[0] * _bc_flag[0], -1 * _extent[1] * _bc_flag[3], _extent[2] * _bc_flag[4]],
                [-1 * _extent[0] * _bc_flag[1], 0., _extent[2] * _bc_flag[4]],
                [0., 0., _extent[2] * _bc_flag[4]],
                [_extent[0] * _bc_flag[0], 0., _extent[2] * _bc_flag[4]],
                [-1 * _extent[0] * _bc_flag[1], _extent[1] * _bc_flag[2], _extent[2] * _bc_flag[4]],
                [0., _extent[1] * _bc_flag[2], _extent[2] * _bc_flag[4]],
                [_extent[0] * _bc_flag[0], _extent[1] * _bc_flag[2], _extent[2] * _bc_flag[4]]
            ]

            '''make scalar array object from above shifts'''
            _tmp_list_local = []
            '''zero scalar array for data that is not position dependent'''
            _tmp_zero = range(26)
            for ix in range(26):
                _tmp_list_local += _cell_shifts[ix]
                _tmp_zero[ix] = 0

            self._cell_shifts_array_pbc = host.Array(_tmp_list_local, dtype=ctypes.c_double)



        else:
            _shift_code = ''
            _args = {}


        _packing_code = '''
        int index = 0;


        //Loop over directions
        for(int ix = 0; ix<26; ix++ ){

            CES[ix] = index;

            //get the start and end indices in the array containing cell indices
            const int start = CCA_I[ix];
            const int end = CCA_I[ix+1];


            //loop over cells
            for(int iy = start; iy < end; iy++){

                // current cell
                int c_i = CIA[iy];

                // first particle
                int iz = q[cell_start+c_i];

                while(iz > -1){

                    // loop over the number of components for particle dat.
                    for(int cx = 0; cx<%(NCOMP)s;cx++){
                        SEND_BUFFER[LINIDX_2D(%(NCOMP)s,index,cx)] = data_buffer[LINIDX_2D(%(NCOMP)s,iz,cx)] %(SHIFT_CODE)s;
                    }
                    index++;
                    iz = q[iz];
                }
            }

        }



        ''' % {'NCOMP': self.ncomp, 'SHIFT_CODE': _shift_code}

        _static_args = {
            'cell_start': ctypes.c_int,  # ctypes.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end])
            'data_buffer': host.pointer_lookup[self.dtype]
        }

        _args['q'] = host.NullIntArray  # cell.cell_list.cell_list.ctypes_data
        _args['CCA_I'] = host.NullIntArray
        _args['CIA'] = host.NullIntArray
        _args['CSA'] = host.NullDoubleArray # self._cell_shifts_array
        _args['SEND_BUFFER'] = host.null_matrix(self.dtype) # packing to send buffer

        self._cumulative_exchange_sizes = host.Array(ncomp=26,dtype=ctypes.c_int)

        _args['CES'] = host.NullIntArray

        _headers = ['stdio.h']
        _kernel = kernel.Kernel('ParticleDatHaloPackingCode', _packing_code, None, _headers, None, _static_args)

        self._halo_packing_lib = build.SharedLib(_kernel, _args)
        self._halo_packing_buffer = host.Matrix(nrow=self.npart, ncol=self.ncomp)

    def halo_pack(self):
        """
        Pack data to prepare for halo exchange.
        """
        if self._halo_packing_lib is None:
            self._setup_halo_packing()

        _boundary_groups_contents_array, _exchange_sizes = halo.HALOS.get_boundary_cell_contents_count

        if _exchange_sizes.dat.sum() > self._halo_packing_buffer.nrow:
            if runtime.VERBOSE.level > 3:
                print "rank:", mpi.MPI_HANDLE.rank, "halo send buffer resized"
            self._halo_packing_buffer.realloc(nrow=_exchange_sizes.dat.sum(), ncol=self.ncomp)

        _static_args = {
            'cell_start': ctypes.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end]),
            'data_buffer': self.ctypes_data
        }

        _dynamic_args = {
                         'q': cell.cell_list.cell_list,
                         'SEND_BUFFER': self._halo_packing_buffer
                         }


        _tmp, self._boundary_groups_start_end_indices = halo.HALOS.get_boundary_cell_groups
        _dynamic_args['CIA'] = _tmp
        _dynamic_args['CCA_I'] = self._boundary_groups_start_end_indices


        _dynamic_args['CES'] = self._cumulative_exchange_sizes

        if self.name == 'positions':
            _dynamic_args['CSA'] = self._cell_shifts_array_pbc


        self._halo_packing_lib.execute(static_args=_static_args, dat_dict=_dynamic_args)

    def _transfer_unpack(self):
        """
        Transfer the packed data. Will use the particle dat as the recv buffer.
        """

        _halo_cell_groups, _halo_groups_start_end_indices = halo.HALOS.get_halo_cell_groups

        # reset halo starting position in dat.
        self.halo_start_reset()

        # Array
        if self._cell_contents_recv is None or halo.HALOS.check_valid is False:
            self._cell_contents_recv = host.Array(ncomp=_halo_cell_groups.ncomp, dtype=ctypes.c_int)

        # zero incoming sizes array
        self._cell_contents_recv.zero()

        _boundary_groups_contents_array, _exchange_sizes = halo.HALOS.get_boundary_cell_contents_count

        _status = mpi.Status()

        # SEND START -------------------------------------------------------------------------------------------
        for i in range(26):
            # Exchange sizes --------------------------------------------------------------------------
            if halo.HALOS.send_ranks[i] > -1 and halo.HALOS.recv_ranks[i] > -1:
                mpi.MPI_HANDLE.comm.Sendrecv(_boundary_groups_contents_array[
                                             self._boundary_groups_start_end_indices[i]:self._boundary_groups_start_end_indices[i + 1]:],
                                             halo.HALOS.send_ranks[i],
                                             halo.HALOS.send_ranks[i],
                                             self._cell_contents_recv[
                                             _halo_groups_start_end_indices[i]:_halo_groups_start_end_indices[i + 1]:],
                                             halo.HALOS.recv_ranks[i],
                                             mpi.MPI_HANDLE.rank,
                                             _status)

            elif halo.HALOS.send_ranks[i] > -1:
                mpi.MPI_HANDLE.comm.Send(_boundary_groups_contents_array[
                                         self._boundary_groups_start_end_indices[i]:self._boundary_groups_start_end_indices[i + 1]:],
                                         halo.HALOS.send_ranks[i],
                                         halo.HALOS.send_ranks[i])

            elif halo.HALOS.recv_ranks[i] > -1:
                mpi.MPI_HANDLE.comm.Recv(self._cell_contents_recv[
                                         _halo_groups_start_end_indices[i]:_halo_groups_start_end_indices[i + 1]:],
                                         halo.HALOS.recv_ranks[i],
                                         mpi.MPI_HANDLE.rank,
                                         _status)




            # Exchange data --------------------------------------------------------------------------
            if halo.HALOS.send_ranks[i] > -1 and halo.HALOS.recv_ranks[i] > -1:
                mpi.MPI_HANDLE.comm.Sendrecv(self._halo_packing_buffer.dat[
                                             self._cumulative_exchange_sizes[i]:self._cumulative_exchange_sizes[i] + _exchange_sizes[i]:,
                                             ::],
                                             halo.HALOS.send_ranks[i],
                                             halo.HALOS.send_ranks[i],
                                             self.dat[self.halo_start::, ::],
                                             halo.HALOS.recv_ranks[i],
                                             mpi.MPI_HANDLE.rank,
                                             _status)

                _shift = _status.Get_count(mpi.mpi_map[self.dtype])
                self.halo_start_shift(_shift / self.ncomp)

            elif halo.HALOS.send_ranks[i] > -1:
                mpi.MPI_HANDLE.comm.Send(self._halo_packing_buffer.dat[
                                         self._cumulative_exchange_sizes[i]:self._cumulative_exchange_sizes[i] + _exchange_sizes[i],
                                         ::],
                                         halo.HALOS.send_ranks[i],
                                         halo.HALOS.send_ranks[i])

            elif halo.HALOS.recv_ranks[i] > -1:

                mpi.MPI_HANDLE.comm.Recv(self.dat[self.halo_start::, ::],
                                         halo.HALOS.recv_ranks[i],
                                         mpi.MPI_HANDLE.rank,
                                         _status)

                _shift = _status.Get_count(mpi.mpi_map[self.dtype])

                self.halo_start_shift(_shift / self.ncomp)

        # SEND END -------------------------------------------------------------------------------------------
        if self.name == 'positions':
            cell.cell_list.sort_halo_cells(_halo_cell_groups, self._cell_contents_recv, self.npart)


















###################################################################################################
# TypedDat.
###################################################################################################


class TypedDat(host.Matrix):
    """
    Base class to hold floating point properties in matrix form of particles based on particle type.

    :arg int nrow: First dimension extent.
    :arg int ncol: Second dimension extent.
    :arg double initial_value: Value to initialise array with, default 0.0.
    :arg str name: Collective name of stored vars eg positions.
    """

    def __init__(self, nrow=1, ncol=1, initial_value=None, name=None, dtype=ctypes.c_double):

        self.name = str(name)
        """:return: Name of TypedDat instance."""
        self.idtype = dtype

        if initial_value is not None:
            if (type(initial_value) is np.ndarray) or type(initial_value) is list:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)


    def __call__(self, mode=access.RW, halo=True):

        return self, mode

    def __getitem__(self, ix):
        return self.dat[ix]

    def __setitem__(self, ix, val):
        self.dat[ix] = val






















