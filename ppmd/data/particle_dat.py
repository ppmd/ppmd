from __future__ import print_function, division, absolute_import
"""
This module contains high level arrays and matrices.
"""

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np

np.set_printoptions(threshold=1000)

# package level
from ppmd import access, mpi, runtime, host, opt
from ppmd.lib import build

from ppmd.data.scalar_array import ScalarArray

_MPI = mpi.MPI
SUM = _MPI.SUM
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier

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


}rst_doc
"""

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

    def __init__(self, npart=0, ncomp=1, initial_value=None, name=None, dtype=ctypes.c_double):
        # version ids. Internal then halo.

        assert ncomp > 0, "Negative number of components is not supported."

        self._vid_int = 0
        self._vid_halo = -1
        self.vid_halo_cell_list = -1

        self.group = None

        # Initialise timers
        self.timer_comm = opt.Timer()
        self.timer_pack = opt.Timer()
        self.timer_transfer = opt.Timer(runtime.TIMER, 0)
        self.timer_transfer_1 = opt.Timer(runtime.TIMER, 0)
        self.timer_transfer_2 = opt.Timer(runtime.TIMER, 0)
        self.timer_transfer_resize = opt.Timer(runtime.TIMER, 0)

        self.name = name
        """:return: The name of the ParticleDat instance."""


        self.idtype = dtype
        self._dat = host._make_array(initial_value=initial_value,
                                      dtype=dtype,
                                      nrow=npart,
                                      ncol=ncomp)

        self.max_npart = self._dat.shape[0]
        """:return: The maximum number of particles which can be stored within
        this particle dat."""

        self.npart_local = self._dat.shape[0]
        """:return: The number of particles with properties stored in the
        particle dat."""

        self.ncomp = self.ncol
        """:return: The number of components stored for each particle."""

        self.halo_start = self.npart_local
        """:return: The starting index of the halo region of the particle dat. """
        self.npart_halo = 0
        self.npart_local_halo = 0
        """:return: The number of particles currently stored within the halo region of the particle dat."""


        self._resize_callback = None
        self._version = 0

        self._exchange_lib = None
        self._tmp_halo_space = host.Array(ncomp=1, dtype=self.dtype)

        # tmp space for norms/maxes etc
        self._norm_tmp = ScalarArray(ncomp=1, dtype=self.dtype)
        self._linf_norm_lib = None


    @property
    def data(self):
        self._vid_int += 1
        return self._dat

    @data.setter
    def data(self, value):
        self._vid_int += 1
        self._dat = value

    def set_val(self, val):
        """
        Set all the entries in the particle dat to the same specified value.

        :param val: Value to set all entries to.
        """
        self.data[..., ...] = val

    def zero(self, n=None):
        if n is None:
            self.data.fill(0)
        else:
            # self.data[:n:, ::] = self.idtype(0.0)
            self.data[:n:, ::] = 0

    def norm_linf(self):
        """
        return the L1 norm of the array
        """

        tmp = np.linalg.norm(
            self.data[:self.npart_local:,:].ravel(),
            np.inf
        )
        return tmp


    def max(self):
        """
        :return: Maximum of local particles
        """
        return self._dat[0:self.npart_local:].max()

    @property
    def npart_total(self):
        """
        Get the total number of particles in the dat including halo particles.

        :return:
        """
        return self.npart_local + self.npart_halo


    def __getitem__(self, ix):
        return self.data[ix]

    def __setitem__(self, ix, val):
        self._vid_int += 1
        self.data[ix] = val

    def __str__(self):
        return str(self.data[::])

    def __repr__(self):
        # self.__str__()
        return "ParticleDat"

    def __call__(self, mode=access.RW, halo=True):

        return self, mode, halo

    def copy(self):
        #print "dat:", self._dat
        return ParticleDat(initial_value=self._dat[0:self.npart_local:],
                           ncomp=self.ncomp,
                           npart=self.npart_local,
                           dtype=self.dtype)



    def broadcast_data_from(self, rank=0, _resize_callback=True):
        # in terms of MPI_COMM_WORLD
        assert (rank>-1) and (rank<_MPISIZE), "Invalid mpi rank"

        if _MPISIZE == 1:
            return
        else:
            s = np.array([self._dat.shape[0]], dtype=ctypes.c_int)
            _MPIWORLD.Bcast(s, root=rank)
            self.resize(s[0], _callback=_resize_callback)
            _MPIWORLD.Bcast(self._dat, root=rank)


    def gather_data_on(self, rank=0, _resize_callback=False):

        # in terms of MPI_COMM_WORLD
        assert (rank>-1) and (rank<_MPISIZE), "Invalid mpi rank"
        if _MPISIZE == 1:
            return
        else:

            counts = _MPIWORLD.gather(self.npart_local, root=rank)

            disp = None
            tmp = np.zeros(1)

            send_size = self.npart_local

            if _MPIRANK == rank:

                self.resize(sum(counts), _callback=_resize_callback)
                self.npart_local = sum(counts)
                disp = [0] + counts[:-1:]
                disp = tuple(np.cumsum(self.ncomp * np.array(disp)))

                counts = tuple([self.ncomp*c for c in counts])


                tmp = np.zeros([self.npart_local, self.ncomp], dtype=self.dtype)


            _MPIWORLD.Gatherv(sendbuf=self._dat[:send_size:,::],
                             recvbuf=(tmp, counts, disp, None),
                             root=rank)

            if _MPIRANK == rank:
                self._dat = tmp



    def ctypes_data_access(self, mode=access.RW, pair=True):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """
        if mode is access.INC0:
            self.zero(self.npart_local)

        exchange = False

        if mode.halo and pair:
            if self.group is not None:
                celllist = self.group.get_cell_to_particle_map()

                if celllist.halos_exist is True and \
                        (self._vid_int > self._vid_halo or
                        self.vid_halo_cell_list < celllist.version_id):

                    exchange = True
            else:
                # halo exchanges are currently not functional wihout a group
                exchange = False

        if exchange:
            self.halo_exchange()

            self._vid_halo = self._vid_int
            self.vid_halo_cell_list = celllist.version_id


        return self._dat.ctypes.data_as(ctypes.POINTER(self.dtype))

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
        self.npart_local_halo = self.halo_start - self.npart_local


    def halo_start_set(self, index):
        """
        Set the start of the halo region in the particle dat to the specified
         index.
        """
        if index < self.npart_local:
            if index >= 0:
                self.npart_local = index
                self.halo_start = index

        else:
            self.halo_start_reset()

        self.npart_halo = 0

    def halo_start_reset(self):
        """
        Reset the starting postion of the halo region in the particle dat to
         the end of the local particles.
        """
        self.halo_start = self.npart_local
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

        if n > self._dat.shape[0]:
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
        #if self.group._cell_to_particle_map.version_id > self.group._cell_to_particle_map.halo_version_id:

        # 0 index contains number of expected particles
        # 1 index contains the required size of tmp arrays

        self.halo_start_reset()

        idi = self.group._cell_to_particle_map.version_id
        idh = self.group._cell_to_particle_map.halo_version_id

        _halo_sizes = self.group._halo_update_exchange_sizes()

        if self._tmp_halo_space.ncomp < (self.ncomp * _halo_sizes[1]):
            #print "\t\t\tresizing temp halo space", _halo_sizes[1]
            self._tmp_halo_space.realloc(int(1.1 * _halo_sizes[1] * self.ncomp))


        self._transfer_unpack()
        self.halo_start_shift(_halo_sizes[0])

        self.group._halo_update_post_exchange()

        self._vid_halo = self._vid_int
        self.timer_comm.pause()

        opt.PROFILE[
            self.__class__.__name__+':'+ self.name +':halo_exchange'
        ] = (self.timer_comm.time())



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

                //for( int dir=0 ; dir<6 ; dir++ ){
                //    cout << "dir: " << dir << " count: " << dir_counts[dir] << endl;;
                //}


                for( int dir=0 ; dir<6 ; dir++ ){
                    //for( int iy=0 ; iy<%(NCOMP)s ; iy++ ){
                    //    cout << "\tdir: " << dir << " comp " << iy << " shift " << SHIFT[dir*%(NCOMP)s + iy] << endl;
                    //}
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

                                //cout << "packed: " << b_tmp[p_index * %(NCOMP)s +iy];

                                #ifdef POS
                                    b_tmp[p_index * %(NCOMP)s + iy] += SHIFT[dir*%(NCOMP)s + iy];
                                #endif

                                //cout << " p_shifted: " << b_tmp[p_index * %(NCOMP)s +iy] << endl;
                            }

                        ix = cell_list[ix];}
                    }

                    /*
                    cout << " SEND | ";
                    for( int tx=0 ; tx < (p_index + 1)*3; tx++){
                        cout << b_tmp[tx] << " |";
                    }
                    cout << endl;
                    */

                    // start the sendrecv as non blocking.
                    if (( SEND_RANKS[dir] > -1 ) && ( p_index > -1 ) ){
                    MPI_Isend((void *) b_tmp, (p_index + 1) * %(NCOMP)s, %(MPI_DTYPE)s,
                             SEND_RANKS[dir], rank, MPI_COMM, &sr);
                    }

                    if (( RECV_RANKS[dir] > -1 ) && ( dir_counts[dir] > 0 ) ){
                    MPI_Irecv((void *) &DAT[DAT_END * %(NCOMP)s], %(NCOMP)s * dir_counts[dir],
                              %(MPI_DTYPE)s, RECV_RANKS[dir], RECV_RANKS[dir], MPI_COMM, &rr);
                    }

                    //cout << "DAT_END: " << DAT_END << endl;



                    int DAT_END_T = DAT_END;
                    DAT_END += dir_counts[dir];

                    // build halo part of cell list whilst exchange occuring.

                    //#ifdef POS
                    if (sort_flag > 0){

                        for( int hxi=h_s ; hxi<h_e ; hxi++ ){

                            // index of a halo cell
                            const int hx = h_arr[ hxi ];

                            // number of particles in cell
                            const int hx_count = ccc[ hx ];

                            if (hx_count > 0) {

                            //cout << "\tsorting cell: " << hx << " ccc: " << hx_count << endl;
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
                    //#endif

                    // after send has completed move to next direction.
                    if (( SEND_RANKS[dir] > -1 ) && ( p_index > -1 ) ){
                        MPI_Wait(&sr, MPI_STATUS_IGNORE);
                    }

                    if (( RECV_RANKS[dir] > -1 ) && ( dir_counts[dir] > 0 ) ){
                        MPI_Wait(&rr, MPI_STATUS_IGNORE);
                    }

                    //MPI_Barrier(MPI_COMM);


                //cout << "dir end " << dir << " -----------" << endl;

                }

                return;
            }
            '''

            if type(self) is PositionDat:
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

        # print "~~~~~~~~~~~~~~~~~~~preparing exxchange"

        comm = self.group.domain.comm
        _h = self.group._halo_manager.get_halo_cell_groups()
        _b = self.group._halo_manager.get_boundary_cell_groups()

        if self.group._cell_to_particle_map.version_id > self.group._cell_to_particle_map.halo_version_id:
            _sort_flag = ctypes.c_int(1)
        else:
            _sort_flag = ctypes.c_int(-1)

        #print "SORT FLAG:", _sort_flag.value, "cell vid:", self.group._cell_to_particle_map.version_id, "halo vid:", self.group._cell_to_particle_map.halo_version_id

        # print str(mpi.MPI_HANDLE.rank) + " -------------- before exchange lib ------------------"
        # sys.stdout.flush()


        # print "HALO_SEND", self.group._halo_manager.get_send_ranks().data
        # print "HALO_RECV", self.group._halo_manager.get_recv_ranks().data

        self._exchange_lib(self.ctypes_data,
                           ctypes.c_int(self.npart_local),
                           self.group._halo_manager.get_position_shifts().ctypes_data,
                           ctypes.c_int(comm.py2f()),
                           self.group._halo_manager.get_send_ranks().ctypes_data,
                           self.group._halo_manager.get_recv_ranks().ctypes_data,
                           _h[1].ctypes_data,
                           _b[1].ctypes_data,
                           _h[0].ctypes_data,
                           _b[0].ctypes_data,
                           self.group._halo_manager.get_dir_counts().ctypes_data,
                           self.group._cell_to_particle_map.offset,
                           _sort_flag,
                           self.group._cell_to_particle_map.cell_contents_count.ctypes_data,
                           self.group._cell_to_particle_map.cell_reverse_lookup.ctypes_data,
                           self.group._cell_to_particle_map.cell_list.ctypes_data,
                           self._tmp_halo_space.ctypes_data
                           )


        # print str(mpi.MPI_HANDLE.rank) +  " --------------- after exchange lib ------------------"
        # print self.npart
        # print self._dat


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

    def remove_particles(self, index=None):
        """
        Remove particles based on host.Array index
        :param index: host.Array with indices to remove
        :return:
        """

        assert index is not None, "No index passed"




#########################################################################
# PositionDat.
#########################################################################

class PositionDat(ParticleDat):
    pass


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

    def __init__(self, nrow=1, ncol=1, initial_value=None, name=None, dtype=ctypes.c_double, key=None):

        assert key is not None, "No key passed to TypedDat"

        self.key = key

        self.name = str(name)
        """:return: Name of TypedDat instance."""
        self.idtype = dtype

        self._dat = host._make_array(initial_value=initial_value,
                                     dtype=dtype,
                                     nrow=nrow,
                                     ncol=ncol)

        self._version = 0


    def __call__(self, mode=access.RW, halo=True):

        return self, mode

    def __getitem__(self, ix):
        return self.data[ix]

    def __setitem__(self, ix, val):
        self.data[ix] = val


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













