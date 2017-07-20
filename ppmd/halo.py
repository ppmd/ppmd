__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


# system level
import ctypes
import numpy as np

# package level
from ppmd import host, mpi, opt, runtime
from ppmd.lib import build


# ===========================================================================

class CellSlice(object):
    def __getitem__(self, item):
        return item

# create a CellSlice object for easier halo definition.
Slice = CellSlice()


def create_halo_pairs_slice_halo(domain_in, slicexyz, direction):
    """
    Automatically create the pairs of cells for halos. Slices through 
    whole domain including halo cells.
    """

    cell_array = domain_in.cell_array
    extent = domain_in.extent
    comm = domain_in.comm
    dims = mpi.cartcomm_dims(comm)
    top = mpi.cartcomm_top(comm)
    periods = mpi.cartcomm_periods(comm)

    xr = range(0, cell_array[0])[slicexyz[0]]
    yr = range(0, cell_array[1])[slicexyz[1]]
    zr = range(0, cell_array[2])[slicexyz[2]]

    if type(xr) is not list:
        xr = [xr]
    if type(yr) is not list:
        yr = [yr]
    if type(zr) is not list:
        zr = [zr]

    l = len(xr) * len(yr) * len(zr)

    b_cells = np.zeros(l, dtype=ctypes.c_int)
    h_cells = np.zeros(l, dtype=ctypes.c_int)

    i = 0

    for iz in zr:
        for iy in yr:
            for ix in xr:
                b_cells[i] = ix + (iy + iz * cell_array[1]) * cell_array[0]

                _ix = (ix + direction[0] * 2) % cell_array[0]
                _iy = (iy + direction[1] * 2) % cell_array[1]
                _iz = (iz + direction[2] * 2) % cell_array[2]

                h_cells[i] = _ix + (_iy + _iz * cell_array[1]) * cell_array[0]

                i += 1

    shift = np.zeros(3, dtype=ctypes.c_double)
    for ix in range(3):
        if top[ix] == 0 and \
                        periods[ix] == 1 and \
                        direction[ix] == -1:

            shift[ix] = extent[ix]

        if top[ix] == dims[ix] - 1 and \
                        periods[ix] == 1 and \
                        direction[ix] == 1:

            shift[ix] = -1. * extent[ix]

    recv_rank = mpi.cartcomm_shift(
        comm,
        (-1 * direction[0], -1 * direction[1], -1 * direction[2])
    )
    send_rank = mpi.cartcomm_shift(comm, direction)

    return b_cells, h_cells, shift, send_rank, recv_rank



class CartesianHaloSix(object):

    def __init__(self, domain_func, cell_to_particle_map):
        self._timer = opt.Timer(runtime.TIMER, 0, start=True)
        
        self._domain_func = domain_func
        self._domain = None

        self._cell_to_particle_map = cell_to_particle_map

        self._ca_copy = [None, None, None]

        self._version = -1

        self._init = False

        # vars init
        self._boundary_cell_groups = host.Array(dtype=ctypes.c_int)
        self._boundary_groups_start_end_indices = host.Array(ncomp=7, dtype=ctypes.c_int)
        self._halo_cell_groups = host.Array(dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = host.Array(ncomp=7, dtype=ctypes.c_int)
        self._boundary_groups_contents_array = host.Array(dtype=ctypes.c_int)
        self._exchange_sizes = host.Array(ncomp=6, dtype=ctypes.c_int)

        self._send_ranks = host.Array(ncomp=6, dtype=ctypes.c_int)
        self._recv_ranks = host.Array(ncomp=6, dtype=ctypes.c_int)

        self._h_count = ctypes.c_int(0)
        self._t_count = ctypes.c_int(0)

        self._h_tmp = host.Array(ncomp=10, dtype=ctypes.c_int)
        self._b_tmp = host.Array(ncomp=10, dtype=ctypes.c_int)

        self.dir_counts = host.Array(ncomp=6, dtype=ctypes.c_int)


        self._halo_shifts = None

        # ensure first update
        self._boundary_cell_groups.inc_version(-1)
        self._boundary_groups_start_end_indices.inc_version(-1)
        self._halo_cell_groups.inc_version(-1)
        self._halo_groups_start_end_indices.inc_version(-1)
        self._boundary_groups_contents_array.inc_version(-1)
        self._exchange_sizes.inc_version(-1)

        self._setup()


        self._exchange_sizes_lib = None
        self._cell_contents_count_tmp = None


    def _setup(self):
        """
        Internally setup the libraries for the calculation of exchange sizes.
        """
        pass

        self._init = True

    def _update_domain(self):
        self._domain = self._domain_func()

    def _get_pairs(self):

        self._update_domain()


        _cell_pairs = (

                # As these are the first exchange the halos cannot contain anything useful
                create_halo_pairs_slice_halo(self._domain, Slice[ 1, 1:-1 ,1:-1],(-1,0,0)),
                create_halo_pairs_slice_halo(self._domain, Slice[-2, 1:-1 ,1:-1 ],(1,0,0)),
                
                # As with the above no point exchanging anything extra in z direction
                create_halo_pairs_slice_halo(self._domain, Slice[::, 1, 1:-1],(0,-1,0)),
                create_halo_pairs_slice_halo(self._domain, Slice[::,-2, 1:-1],(0,1,0)),

                # Exchange all halo cells from x and y
                create_halo_pairs_slice_halo(self._domain, Slice[::,::,1],(0,0,-1)),
                create_halo_pairs_slice_halo(self._domain, Slice[::,::,-2],(0,0,1))
            )

        _bs = np.zeros(1, dtype=ctypes.c_int)
        _b = np.zeros(0, dtype=ctypes.c_int)

        _hs = np.zeros(1, dtype=ctypes.c_int)
        _h = np.zeros(0, dtype=ctypes.c_int)

        _s = np.zeros(0, dtype=ctypes.c_double)

        _len_h_tmp = 10
        _len_b_tmp = 10

        for hx, bhx in enumerate(_cell_pairs):

            # print hx, bhx


            _len_b_tmp = max(_len_b_tmp, len(bhx[0]))
            _len_h_tmp = max(_len_h_tmp, len(bhx[1]))

            # Boundary and Halo start index.
            _bs = np.append(_bs, ctypes.c_int(len(bhx[0]) + _bs[-1] ))
            _hs = np.append(_hs, ctypes.c_int(len(bhx[1]) + _hs[-1] ))

            # Actual cell indices
            _b = np.append(_b, bhx[0])
            _h = np.append(_h, bhx[1])

            # Offset shifts for periodic boundary
            _s = np.append(_s, bhx[2])

            self._send_ranks[hx] = bhx[3]
            self._recv_ranks[hx] = bhx[4]

        if _len_b_tmp > self._b_tmp.ncomp:
            self._b_tmp.realloc(_len_b_tmp)

        if _len_h_tmp > self._h_tmp.ncomp:
            self._h_tmp.realloc(_len_h_tmp)


        # indices in array of cell indices
        self._boundary_groups_start_end_indices = host.Array(_bs, dtype=ctypes.c_int)
        self._halo_groups_start_end_indices = host.Array(_hs, dtype=ctypes.c_int)

        # cell indices
        self._boundary_cell_groups = host.Array(_b, dtype=ctypes.c_int)
        self._halo_cell_groups = host.Array(_h, dtype=ctypes.c_int)

        # shifts for each direction.
        self._halo_shifts = host.Array(_s, dtype=ctypes.c_double)

        self._version = self._domain.cell_array.version


    def get_boundary_cell_groups(self):
        """
        Get the local boundary cells to pack for each halo. Formatted as an
        host.Array. Cells for halo 0 first followed by cells for halo 1 etc.
        Also returns an data.Array of 27 elements with the
        starting positions of each halo within the previous array.

        :return: Tuple, array of local cell indices to pack, array of starting
        points within the first array.
        """
        self._update_domain()
        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._boundary_cell_groups, self._boundary_groups_start_end_indices

    def get_halo_cell_groups(self):
        """
        Get the local halo cells to unpack into for each halo. Formatted as an
        cuda_base.Array. Cells for halo 0 first followed by cells for halo 1
        etc. Also returns an data.Array of 27 elements with the starting
        positions of each halo within the previous array.

        :return: Tuple, array of local halo cell indices to unpack into, array
        of starting points within the first array.
        """
        self._update_domain()
        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._halo_cell_groups, self._halo_groups_start_end_indices

    def get_boundary_cell_contents_count(self):
        """
        Get the number of particles in the corresponding cells for each halo.
        These are needed such that the cell list can be created without
        inspecting the positions of recvd particles.

        :return: Tuple: Cell contents count for each cell in same order as
        local boundary cell list, Exchange sizes for each halo.
        """
        if not self._init:
            print("cuda_halo.CartesianHalo error. Library not initalised, " \
                  "this error means the internal setup failed.")
            quit()

        self._exchange_sizes.zero()

        return self._boundary_groups_contents_array, self._exchange_sizes

    def get_position_shifts(self):
        """
        Calculate flag to determine if a boundary between processes is also
        a boundary in domain.
        """
        self._update_domain()
        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._halo_shifts

    def get_send_ranks(self):
        """
        Get the mpi ranks to send to.
        """
        self._update_domain()
        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._send_ranks

    def get_recv_ranks(self):
        """
        Get the mpi ranks to recv from.
        """
        self._update_domain()
        if self._version < self._domain.cell_array.version:
            self._get_pairs()

        return self._recv_ranks

    def get_dir_counts(self):
        return self.dir_counts

    def exchange_cell_counts(self):
        """
        Exchange the contents count of cells between processes. This is
        provided as a method in halo to avoid repeated exchanging of cell
        occupancy counts if multiple ParticleDat objects are being
        communicated.
        """
        self._update_domain()
        if self._exchange_sizes_lib is None:

            _es_args = '''
            const int f_MPI_COMM,             // F90 comm from mpi4py
            const int * RESTRICT SEND_RANKS,  // send directions
            const int * RESTRICT RECV_RANKS,  // recv directions
            const int * RESTRICT h_ind,       // halo indices
            const int * RESTRICT b_ind,       // local b indices
            const int * RESTRICT h_arr,       // h cell indices
            const int * RESTRICT b_arr,       // b cell indices
            int * RESTRICT ccc,               // cell contents count
            int * RESTRICT h_count,           // number of halo particles
            int * RESTRICT t_count,           // amount of tmp space needed
            int * RESTRICT h_tmp,             // tmp space for recving
            int * RESTRICT b_tmp,             // tmp space for sending
            int * RESTRICT dir_counts         // expected recv counts
            '''

            _es_header = '''
            #include <generic.h>
            #include <mpi.h>
            #include <iostream>
            using namespace std;
            #define RESTRICT %(RESTRICT)s

            extern "C" void HALO_ES_LIB(%(ARGS)s);
            '''

            _es_code = '''

            void HALO_ES_LIB(%(ARGS)s){
                *h_count = 0;
                *t_count = 0;

                // get mpi comm and rank
                MPI_Comm MPI_COMM = MPI_Comm_f2c(f_MPI_COMM);
                int rank = -1; MPI_Comm_rank( MPI_COMM, &rank );
                MPI_Status MPI_STATUS;

                // [W E] [N S] [O I]
                for( int dir=0 ; dir<6 ; dir++ ){

                    //cout << "dir " << dir << "-------" << endl;

                    const int dir_s = b_ind[dir];             // start index
                    const int dir_c = b_ind[dir+1] - dir_s;   // cell count

                    const int dir_s_r = h_ind[dir];             // start index
                    const int dir_c_r = h_ind[dir+1] - dir_s_r; // cell count

                    int tmp_count = 0;
                    for( int ix=0 ; ix<dir_c ; ix++ ){
                        b_tmp[ix] = ccc[b_arr[dir_s + ix]];    // copy into
                                                               // send buffer

                        tmp_count += ccc[b_arr[dir_s + ix]];
                    }

                    *t_count = MAX(*t_count, tmp_count);


                    if(rank == RECV_RANKS[dir]){

                        for( int tx=0 ; tx < dir_c ; tx++ ){
                            h_tmp[tx] = b_tmp[tx];
                        }

                    } else {
                    MPI_Sendrecv ((void *) b_tmp, dir_c, MPI_INT,
                                  SEND_RANKS[dir], rank,
                                  (void *) h_tmp, dir_c_r, MPI_INT,
                                  RECV_RANKS[dir], RECV_RANKS[dir],
                                  MPI_COMM, &MPI_STATUS);
                    }

                    tmp_count=0;
                    for( int ix=0 ; ix<dir_c_r ; ix++ ){
                        ccc[h_arr[dir_s_r + ix]] = h_tmp[ix];
                        *h_count += h_tmp[ix];
                        tmp_count += h_tmp[ix];
                    }
                    dir_counts[dir] = tmp_count;
                    *t_count = MAX(*t_count, tmp_count);

                }

                return;
            }
            '''

            _es_dict = {'ARGS': _es_args,
                        'RESTRICT': build.MPI_CC.restrict_keyword}

            _es_header %= _es_dict
            _es_code %= _es_dict

            self._exchange_sizes_lib = build.simple_lib_creator(_es_header,
                                                                _es_code,
                                                                'HALO_ES_LIB',
                                                                CC=build.MPI_CC
                                                                )['HALO_ES_LIB']

        # End of creation code -----------------------------------------------

        # update internal arrays
        if self._version < self._domain.cell_array.version:
            self._get_pairs()


        ccc = self._cell_to_particle_map.cell_contents_count

        # This if allows the host size exchnage code to be used for the gpu
        if type(ccc) is host.Array:
            ccc_ptr = ccc.ctypes_data

        else:
            if self._cell_contents_count_tmp is None:
                self._cell_contents_count_tmp = host.Array(ncomp=ccc.ncomp, dtype=ctypes.c_int)
            elif self._cell_contents_count_tmp.ncomp < ccc.ncomp:
                self._cell_contents_count_tmp.realloc(ccc.ncomp)

            #make a local copy of the cell contents counts
            self._cell_contents_count_tmp[:] = ccc[:]
            ccc_ptr = self._cell_contents_count_tmp.ctypes_data


        assert ccc_ptr is not None, "No valid Cell Contents Count pointer found."

        self._exchange_sizes_lib(ctypes.c_int(self._domain.comm.py2f()),
                                 self._send_ranks.ctypes_data,
                                 self._recv_ranks.ctypes_data,
                                 self._halo_groups_start_end_indices.ctypes_data,
                                 self._boundary_groups_start_end_indices.ctypes_data,
                                 self._halo_cell_groups.ctypes_data,
                                 self._boundary_cell_groups.ctypes_data,
                                 ccc_ptr,
                                 ctypes.byref(self._h_count),
                                 ctypes.byref(self._t_count),
                                 self._h_tmp.ctypes_data,
                                 self._b_tmp.ctypes_data,
                                 self.dir_counts.ctypes_data)

        # copy new sizes back to original array (eg for gpu)
        if type(ccc) is not host.Array:
            ccc[:] = self._cell_contents_count_tmp[:ccc.ncomp:]


        return self._h_count.value, self._t_count.value


























