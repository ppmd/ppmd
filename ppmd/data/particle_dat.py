"""
This module contains the `ParticleDat` class for adding data to particles.
"""

from __future__ import print_function, division, absolute_import

import ppmd.opt
import ppmd.runtime



__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np


# package level
from ppmd import access, mpi, runtime, host, opt
from ppmd.lib import build

from ppmd.data.scalar_array import ScalarArray
from ppmd.data.data_movement import ParticleDatModifier


SUM = mpi.MPI.SUM

##"""
##rst_doc{
##
##.. contents:
##
##data Module
##===========
##
##.. automodule:: data
##
##ScalarArray
##~~~~~~~~~~~
##
##The class :class:`~data.ScalarArray` is a generic one dimensional array that should be used
##to store data within simulations that is not associated with any particular particles. For
##example the kinetic energy of the system or the array used to bin values when calculating a
##radial distribution.
##
##.. autoclass:: data.ScalarArray
##    :show-inheritance:
##    :undoc-members:
##    :members:
##
##
##ParticleDat
##~~~~~~~~~~~
##
##This classes should be considered as a two dimensional matrix with each row storing the properties
##of a particle. The order of rows in relation to which particle they correspond to should always
##be conserved. This is the default behaviour of any sorting methods implemented in this framework.
##
##.. autoclass:: data.ParticleDat
##    :show-inheritance:
##    :undoc-members:
##    :members:
##
##
##}rst_doc
##"""

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
        self._halo_exchange_count = 0

        self.group = None

        # Initialise timers
        self.timer_comm = ppmd.opt.Timer()
        self.timer_pack = ppmd.opt.Timer()
        self.timer_transfer = ppmd.opt.Timer(runtime.TIMER, 0)
        self.timer_transfer_1 = ppmd.opt.Timer(runtime.TIMER, 0)
        self.timer_transfer_2 = ppmd.opt.Timer(runtime.TIMER, 0)
        self.timer_transfer_resize = ppmd.opt.Timer(runtime.TIMER, 0)

        self.name = name
        """:return: The name of the ParticleDat instance."""

        self.idtype = dtype
        self._dat = host._make_array(
            initial_value=initial_value,
            dtype=dtype,
            nrow=npart,
            ncol=ncomp
        )

        self._ptr = None
        self._ptr_count = 0

        self.max_npart = self._dat.shape[0]
        """:return: The maximum number of particles which can be stored within this particle dat."""

        self.npart_local = self._dat.shape[0]
        """:return: The number of particles with properties stored in the particle dat."""

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

        # default comm is world
        self.comm = mpi.MPI.COMM_WORLD

        self._particle_dat_modifier = ParticleDatModifier(self, type(self) == PositionDat)


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
            self.data[:self.npart_local:, :].ravel(),
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
    
    def mark_halos_old(self):
        self._vid_int += 1

    def modify_view(self):
        return self._particle_dat_modifier

    def sync_view_to_data(self):
        # on devices where the actual data is separate to the view data
        # this is a syncrohisation call
        pass

    @property
    def view(self):
        return self._dat[:self.npart_local:, :].view()

    @property
    def data(self):
        self.mark_halos_old()
        return self._dat

    @data.setter
    def data(self, value):
        self.mark_halos_old()
        self._dat = value
        self._ptr = None

    def __getitem__(self, ix):
        return np.copy(self._dat[ix])

    def __setitem__(self, ix, val):
        self.mark_halos_old()
        self.data[ix] = val
        if type(self) is PositionDat and self.group is not None:
            self.group.invalidate_lists()

    def __str__(self):
        return str(self.data[::])

    def __repr__(self):
        # self.__str__()
        return "ParticleDat_" + str(self.dtype)

    def __call__(self, mode=access.RW, halo=True):

        return self, mode, halo

    def copy(self):
        # print "dat:", self._dat
        return ParticleDat(initial_value=self._dat[0:self.npart_local:],
                           ncomp=self.ncomp,
                           npart=self.npart_local,
                           dtype=self.dtype)

    def broadcast_data_from(self, rank=0, _resize_callback=True):
        # in terms of MPI_COMM_WORLD
        assert (rank > -1) and (rank < self.comm.Get_size()), "Invalid mpi rank"

        if self.comm.Get_size() == 1:
            return
        else:
            s = np.array([self._dat.shape[0]], dtype=ctypes.c_int)
            self.comm.Bcast(s, root=rank)
            self.resize(s[0], _callback=_resize_callback)
            self.comm.Bcast(self._dat, root=rank)

    def gather_data_on(self, rank=0, _resize_callback=False):

        # in terms of MPI_COMM_WORLD
        assert (rank > -1) and (rank < self.comm.Get_size()), "Invalid mpi rank"
        if self.comm.Get_size() == 1:
            return
        else:

            counts = self.comm.gather(self.npart_local, root=rank)

            disp = None
            tmp = np.zeros(1)

            send_size = self.npart_local

            if self.comm.Get_rank() == rank:

                self.resize(sum(counts), _callback=_resize_callback)
                self.npart_local = sum(counts)
                disp = [0] + counts[:-1:]
                disp = tuple(np.cumsum(self.ncomp * np.array(disp)))

                counts = tuple([self.ncomp * c for c in counts])

                tmp = np.zeros([self.npart_local, self.ncomp], dtype=self.dtype)

            self.comm.Gatherv(
                sendbuf=self._dat[:send_size:, ::],
                recvbuf=(tmp, counts, disp, None),
                root=rank
            )

            if self.comm.Get_rank() == rank:
                self._dat = tmp
                self._ptr = None

    def ctypes_data_access(self, mode=access.RW, pair=True):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """

        # if this is being launched in single particle mode we need to extract the access descriptor
        # and the particle ids
        if isinstance(mode, access._ParticleSetAccess):
            local_ids = mode.local_ids
            mode = mode.mode
        else:
            local_ids = None


        if mode is access.INC0:
            if local_ids is None:
                self.zero(self.npart_local)
            else:
                self.data[local_ids, :] = 0

        exchange = False

        if mode.halo and pair:
            if self.group is not None:
                celllist = self.group.get_cell_to_particle_map()

                if celllist.halos_exist is True and \
                        (
                            self._vid_int > self._vid_halo or
                            self.vid_halo_cell_list < celllist.version_id
                        ):

                    exchange = True
            else:
                # halo exchanges are currently not functional wihout a group
                exchange = False

        if exchange:
            self.halo_exchange()

            self._vid_halo = self._vid_int
            self.vid_halo_cell_list = celllist.version_id
        
        if self._ptr is None:
            #self._ptr = self._dat.ctypes.data_as(ctypes.POINTER(self.dtype))
            self._ptr = self._dat.ctypes.get_as_parameter()

        self._ptr_count += 1
        if self._ptr_count % 100 == 0:
            # Check that NumPy has not swapped out the buffer for some reason (or we failed to 
            # get the pointer on a realloc).
            assert self._ptr.value == self._dat.ctypes.get_as_parameter().value

        return self._ptr


    def ctypes_data_post(self, mode=access.RW):
        """
        Call after excuting a method on the data.
        :arg access mode: Access type required by the calling method.
        """

        # if this is being launched in single particle mode we need to extract the access descriptor
        # and the particle ids
        if isinstance(mode, access._ParticleSetAccess):
            local_ids = mode.local_ids
            mode = mode.mode

        if mode.write:
            self.mark_halos_old()

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
        Perform a halo exchange for the particle dat.
        """

        self.timer_comm.start()

        # can only exchage sizes if needed.

        self.halo_start_reset()

        _halo_sizes = self.group._halo_update_exchange_sizes()
        if self._tmp_halo_space.ncomp < (self.ncomp * _halo_sizes[1]):
            # print "\t\t\tresizing temp halo space", _halo_sizes[1]
            self._tmp_halo_space.realloc(int(1.1 * _halo_sizes[1] * self.ncomp))

        self._transfer_unpack()
        self.halo_start_shift(_halo_sizes[0])

        self.group._halo_update_post_exchange()

        self._vid_halo = self._vid_int
        self.timer_comm.pause()
        self._halo_exchange_count += 1

        opt.PROFILE[
            self.__class__.__name__ + ':' + self.name + ':halo_exchange'
        ] = (self.timer_comm.time())
        opt.PROFILE[
            self.__class__.__name__ + ':' + self.name + ':halo_exchange:count'
        ] = (self._halo_exchange_count)

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
            int * RESTRICT cell_linked_list,  // cell list
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
                        int ix = cell_linked_list[cell_offset + ci];
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

                        ix = cell_linked_list[ix];}
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
                                cell_linked_list[cell_offset + hx] = DAT_END_T;

                                for( int iy=0 ; iy<(hx_count-1) ; iy++ ){

                                    cell_linked_list[ DAT_END_T+iy ] = DAT_END_T + iy + 1;
                                    crl[ DAT_END_T+iy ] = hx;

                                }

                                cell_linked_list[ DAT_END_T + hx_count - 1 ] = -1;
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

        comm = self.group.domain.comm
        _h = self.group._halo_manager.get_halo_cell_groups()
        _b = self.group._halo_manager.get_boundary_cell_groups()

        if self.group._cell_to_particle_map.version_id > self.group._cell_to_particle_map.halo_version_id:
            _sort_flag = ctypes.c_int(1)
        else:
            _sort_flag = ctypes.c_int(-1)

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

#########################################################################
# PositionDat.
#########################################################################


class PositionDat(ParticleDat):
    def __init__(self, npart=0, ncomp=3, initial_value=None, name=None, dtype=ctypes.c_double):
        if ncomp != 3: raise RuntimeError('ncomp must be 3 for PositionDat')
        if dtype != ctypes.c_double: raise RuntimeError('dtype must be ctypes.c_double for PositionDat')
        super().__init__(npart=npart, ncomp=3, dtype=ctypes.c_double)

# this needs deleting after cell by cell is updated
class TypedDat(object):
    pass


