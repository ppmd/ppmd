"""
This module contains high level arrays and matrices.
"""
import sys
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

        # TODO: remove
        self._cuda_dat = None

        self._version = 0

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

    def zero(self):
        self.dat.fill(0)

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




###############################################################################
# Blank arrays.
###############################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
"""Empty integer :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""


NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)
"""Empty double :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""

###############################################################################
# ParticleDat.
###############################################################################


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


        # Initialise timers
        self.timer_comm = runtime.Timer(runtime.TIMER, 0)
        self.timer_pack = runtime.Timer(runtime.TIMER, 0)
        self.timer_transfer = runtime.Timer(runtime.TIMER, 0)
        self.timer_transfer_1 = runtime.Timer(runtime.TIMER, 0)
        self.timer_transfer_2 = runtime.Timer(runtime.TIMER, 0)
        self.timer_transfer_resize = runtime.Timer(runtime.TIMER, 0)


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


        self.halo_start = self.npart
        """:return: The starting index of the halo region of the particle dat. """

        self.npart_halo = 0
        """:return: The number of particles currently stored within the halo region of the particle dat."""


        self._resize_callback = None
        self._version = 0

        self._exchange_lib = None
        self._tmp_halo_space = host.Array(ncomp=1, dtype=self.dtype)


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

    def zero(self, n=None):
        if n is None:
            self.dat.fill(0)
        else:
            self.dat[:n:, ::] = self.idtype(0.0)
            


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

        return self, mode, halo

    def ctypes_data_access(self, mode=access.RW):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """
        if mode.read:
            if (self._vid_int > self._vid_halo) and cell.cell_list.halos_exist is True:
                #print "halo exchangeing", self.name
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
        Shift the starting point of the halo in the particle dat by the
        specified shift.
        :param int shift: Offset to shift by.
        """

        self.halo_start += shift
        self.npart_halo = self.halo_start - self.npart

    def halo_start_set(self, index):
        """
        Set the start of the halo region in the particle dat to the specified
         index.
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
        Reset the starting postion of the halo region in the particle dat to
         the end of the local particles.
        """
        self.halo_start = self.npart
        self.npart_halo = 0

    def resize(self, n, _callback=True):
        """
        Resize particle dat to be at least a certain size, does not resize if
        already large enough.
        :arg int n: New minimum size.
        """

        if _callback and (self._resize_callback is not None):
            self._resize_callback(n)
            return
            #pass


        if n > self.max_npart:
            self.max_npart = n
            self.realloc(n, self.ncol)


    def halo_exchange(self):
        """
        Perform a halo exchange for the particle dat. WIP currently only
        functional for positions.
        """

        self.timer_comm.start()




        # new start
        # can only exchage sizes if needed.
        #if cell.cell_list.version_id > cell.cell_list.halo_version_id:

        # 0 index contains number of expected particles
        # 1 index contains the required size of tmp arrays

        self.halo_start_reset()
        _sizes = halo.HALOS.exchange_cell_counts()

        print "\t\tAFTER sizes exchange"

        _size = self.halo_start + _sizes[0]
        self.resize(_size)

        sys.stdout.flush()

        print "\t\tAFTER dat resize"
        if self._tmp_halo_space.ncomp < (self.ncomp * _sizes[1]):
            print "\t\t\tresizing tmp space", _sizes[1]
            self._tmp_halo_space.realloc(_sizes[1] * self.ncomp)

        print "\t\tAFTER TMP resize"


        if (self.name == 'positions') and cell.cell_list.version_id > cell.cell_list.halo_version_id:
            cell.cell_list.prepare_halo_sort(_size)

        print "\t\tAFTER cell resize"

        self._transfer_unpack()
        self.halo_start_shift(_sizes[0])

        if (self.name == 'positions') and cell.cell_list.version_id > cell.cell_list.halo_version_id:
            cell.cell_list.post_halo_exchange()



        self._vid_halo = self._vid_int
        self.timer_comm.pause()


    def _transfer_unpack(self):
        """
        pack and transfer the particle dat, rebuild cell list if needed
        """
        if self._exchange_lib is None:

            _ex_args = '''
            %(DTYPE)s * RESTRICT DAT,         // DAT pointer
            int DAT_END,                      // end of dat.
            const double * RESTRICT SHIFT,    // position shifts
            const int f_MPI_COMM,             // F90 comm from mpi4py
            const int * RESTRICT SEND_RANKS,  // send directions
            const int * RESTRICT RECV_RANKS,  // recv directions
            const int * RESTRICT h_ind,       // halo indices
            const int * RESTRICT b_ind,       // local b indices
            const int * RESTRICT h_arr,       // h cell indices
            const int * RESTRICT b_arr,       // b cell indices
            const int * RESTRICT dir_counts,  // expected recv counts
            const int cell_offset,            // offset for cell list
            const int sort_flag,              // does the cl require updating
            int * RESTRICT ccc,               // cell contents count
            int * RESTRICT crl,               // cell reverse lookup
            int * RESTRICT cell_list,         // cell list
            %(DTYPE)s * RESTRICT b_tmp        // tmp space for sending
            ''' % {'DTYPE': host.ctypes_map[self.dtype]}

            _ex_header = '''
            #include <generic.h>
            #include <mpi.h>
            #include <iostream>
            using namespace std;
            #define RESTRICT %(RESTRICT)s

            %(POS_ENABLE)s

            extern "C" void HALO_EXCHANGE_PD(%(ARGS)s);
            '''

            _ex_code = '''

            void HALO_EXCHANGE_PD(%(ARGS)s){

                // get mpi comm and rank
                MPI_Comm MPI_COMM = MPI_Comm_f2c(f_MPI_COMM);
                int rank = -1; MPI_Comm_rank( MPI_COMM, &rank );
                MPI_Status MPI_STATUS;
                MPI_Request sr;
                MPI_Request rr;

                for( int dir=0 ; dir<6 ; dir++ ){
                    cout << "dir: " << dir << " count: " << dir_counts[dir] << endl;;
                }


                for( int dir=0 ; dir<6 ; dir++ ){
                    for( int iy=0 ; iy<%(NCOMP)s ; iy++ ){
                        cout << "\tdir: " << dir << " comp " << iy << " shift " << SHIFT[dir*%(NCOMP)s + iy] << endl;
                    }
                    const int b_s = b_ind[dir];
                    const int b_e = b_ind[dir+1];
                    const int b_c = b_e - b_s;

                    const int h_s = h_ind[dir];
                    const int h_e = h_ind[dir+1];
                    const int h_c = h_e - h_s;

                    //packing index;
                    int p_index = -1;

                    // packing loop
                    for( int cx=0 ; cx<b_c ; cx++ ){

                        // cell index
                        const int ci = b_arr[b_s + cx];

                        // loop over contents of cell.
                        int ix = cell_list[cell_offset + ci];
                        while(ix > -1){

                            p_index ++;
                            for( int iy=0 ; iy<%(NCOMP)s ; iy++ ){

                                b_tmp[p_index * %(NCOMP)s + iy] = DAT[ix*%(NCOMP)s + iy];

                                cout << "packed: " << b_tmp[p_index * %(NCOMP)s +iy];

                                #ifdef POS
                                    b_tmp[p_index * %(NCOMP)s + iy] += SHIFT[dir*%(NCOMP)s + iy];
                                #endif

                                cout << " p_shifted: " << b_tmp[p_index * %(NCOMP)s +iy] << endl;
                            }

                        ix = cell_list[ix];}
                    }

                    cout << " SEND | ";
                    for( int tx=0 ; tx < (p_index + 1)*3; tx++){
                        cout << b_tmp[tx] << " |";
                    }
                    cout << endl;


                    // start the sendrecv as non blocking.
                    if (( SEND_RANKS[dir] > -1 ) && ( p_index > -1 ) ){
                    MPI_Isend((void *) b_tmp, (p_index + 1) * %(NCOMP)s, %(MPI_DTYPE)s,
                             SEND_RANKS[dir], rank, MPI_COMM, &sr);
                    }

                    if (( RECV_RANKS[dir] > -1 ) && ( dir_counts[dir] > 0 ) ){
                    MPI_Irecv((void *) &DAT[DAT_END * %(NCOMP)s], %(NCOMP)s * dir_counts[dir],
                              %(MPI_DTYPE)s, RECV_RANKS[dir], RECV_RANKS[dir], MPI_COMM, &rr);
                    }

                    cout << "DAT_END: " << DAT_END << endl;



                    int DAT_END_T = DAT_END;
                    DAT_END += dir_counts[dir];

                    // build halo part of cell list whilst exchange occuring.

                    #ifdef POS
                    if (sort_flag > 0){

                        for( int hxi=h_s ; hxi<h_e ; hxi++ ){

                            // index of a halo cell
                            const int hx = h_arr[ hxi ];

                            // number of particles in cell
                            const int hx_count = ccc[ hx ];

                            if (hx_count > 0) {

                            cout << "\tsorting cell: " << hx << " ccc: " << hx_count << endl;
                                cell_list[cell_offset + hx] = DAT_END_T;

                                for( int iy=0 ; iy<(hx_count-1) ; iy++ ){

                                    cell_list[ DAT_END_T+iy ] = DAT_END_T + iy + 1;
                                    crl[ DAT_END_T+iy ] = hx;

                                }

                                cell_list[ DAT_END_T + hx_count - 1 ] = -1;
                                crl[ DAT_END_T + hx_count -1 ] = hx;

                                DAT_END_T += hx_count;
                            }
                        }

                    }
                    #endif

                    // after send has completed move to next direction.
                    if (( SEND_RANKS[dir] > -1 ) && ( p_index > -1 ) ){
                        MPI_Wait(&sr, &MPI_STATUS);
                    }

                    if (( RECV_RANKS[dir] > -1 ) && ( dir_counts[dir] > 0 ) ){
                        MPI_Wait(&rr, &MPI_STATUS);
                    }

                    MPI_Barrier(MPI_COMM);


                cout << "dir end " << dir << " -----------" << endl;

                }

                return;
            }
            '''

            if self.name == 'positions':
                _pos_enable = '#define POS'
            else:
                _pos_enable = ''


            _ex_dict = {'ARGS': _ex_args,
                        'RESTRICT': build.MPI_CC.restrict_keyword,
                        'DTYPE': host.ctypes_map[self.dtype],
                        'POS_ENABLE': _pos_enable,
                        'NCOMP': self.ncomp,
                        'MPI_DTYPE': host.mpi_type_map[self.dtype]}

            _ex_header %= _ex_dict
            _ex_code %= _ex_dict

            self._exchange_lib = build.simple_lib_creator(_ex_header,
                                                          _ex_code,
                                                          'HALO_EXCHANGE_PD',
                                                          CC=build.MPI_CC
                                                          )['HALO_EXCHANGE_PD']


        # End of creation code -----------------------------------------

        print "~~~~~~~~~~~~~~~~~~~preparing exxchange"

        _h = halo.HALOS.get_halo_cell_groups()
        _b = halo.HALOS.get_boundary_cell_groups()

        if (self.name == 'positions') and cell.cell_list.version_id > cell.cell_list.halo_version_id:
            _sort_flag = ctypes.c_int(1)
        else:
            _sort_flag = ctypes.c_int(-1)

        print "SORT FLAG:", _sort_flag.value, "cell vid:", cell.cell_list.version_id, "halo vid:", cell.cell_list.halo_version_id

        print str(mpi.MPI_HANDLE.rank) + " -------------- before exchange lib ------------------"
        sys.stdout.flush()

        self._exchange_lib(self.ctypes_data,
                           ctypes.c_int(self.npart),
                           halo.HALOS.get_position_shifts().ctypes_data,
                           ctypes.c_int(mpi.MPI_HANDLE.fortran_comm),
                           halo.HALOS.get_send_ranks().ctypes_data,
                           halo.HALOS.get_recv_ranks().ctypes_data,
                           _h[1].ctypes_data,
                           _b[1].ctypes_data,
                           _h[0].ctypes_data,
                           _b[0].ctypes_data,
                           halo.HALOS.get_dir_counts().ctypes_data,
                           cell.cell_list.offset,
                           _sort_flag,
                           cell.cell_list.cell_contents_count.ctypes_data,
                           cell.cell_list.cell_reverse_lookup.ctypes_data,
                           cell.cell_list.cell_list.ctypes_data,
                           self._tmp_halo_space.ctypes_data
                           )


        print str(mpi.MPI_HANDLE.rank) +  " --------------- after exchange lib ------------------"
        print self.npart
        print self._dat


        '''
        %(DTYPE)s * RESTRICT DAT,         // DAT pointer
        int DAT_END,                      // end of dat.
        const double * RESTRICT SHIFT,    // position shifts
        const int f_MPI_COMM,             // F90 comm from mpi4py
        const int * RESTRICT SEND_RANKS,  // send directions
        onst int * RESTRICT RECV_RANKS,  // recv directions
        const int * RESTRICT h_ind,       // halo indices
        const int * RESTRICT b_ind,       // local b indices
        const int * RESTRICT h_arr,       // h cell indices
        const int * RESTRICT b_arr,       // b cell indices
        const int * RESTRICT dir_counts,  // expected recv counts
        const int cell_offset,            // offset for cell list
        const int sort_flag,              // does the cl require updating
        int * RESTRICT ccc,               // cell contents count
        int * RESTRICT crl,               // cell reverse lookup
        int * RESTRICT cell_list,         // cell list
        int * RESTRICT b_tmp              // tmp space for sending
        '''



#########################################################################
# TypedDat.
#########################################################################

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


########################################################################
# Type
########################################################################
class Type(object):
    """
    Object to store information such as number of 
    particles in a type.
    """
    def __init__(self):
        self._n = 0
        self._h_n = 0
        self._dats = []

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        # place code to call resize on dats here.
        self._n = int(val)
    
    @property
    def _hn(self):
        return self._h_n

    @_hn.setter
    def _hn(self, val):
        self._h_n = int(val)

    @property
    def _total_size(self):
        return self._h_n + self._n

    def _append_dat(self, dat=None):
        assert dat is not None, "No dat added to Type instance"
        self._dats.append(dat)




















