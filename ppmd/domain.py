

# system level
import numpy as np
import math
import ctypes

# package level
import data
import host
import kernel
import build
import mpi
import runtime
import pio





##############################################################################################################
# BASE DOMAIN HALO
##############################################################################################################


class BaseDomainHalo(object):
    """
    A cell based domain for mpi/private memory. Creates a shell of halos cells around
    each processes internal cells as halos.

    """

    def __init__(self, extent=np.array([1., 1., 1.]), cell_count=1, periods=(1, 1, 1)):

        self._periods = periods

        self._extent = data.ScalarArray(extent)

        self._extent_global = data.ScalarArray(extent)

        self._cell_array = data.ScalarArray(np.array([1, 1, 1]), dtype=ctypes.c_int)
        self._cell_edge_lengths = data.ScalarArray(np.array([1., 1., 1.], dtype=ctypes.c_double))

        self._halos = True

        #vars to return boudary cells
        self._boundary_cell_version = -1
        self._boundary_cells = None

    def get_boundary_cells(self):
        """
        Return a host.Array containing the boundary cell indices of the domain.
        """

        if self._boundary_cell_version < self._cell_array.version:
            _ca = self._cell_array
            _count = (_ca[0] - 2) * (_ca[1] - 2) * (_ca[2] - 2) - (_ca[0] - 4) * (_ca[1] - 4) * (_ca[2] - 4)

            self._boundary_cells = host.Array(ncomp=_count, dtype=ctypes.c_int)
            m = 0

            for ix in range(1, _ca[0] - 1):
                for iy in range(1, _ca[1] - 1):

                    self._boundary_cells[m] = ix + _ca[0]*(iy + _ca[1])
                    self._boundary_cells[m + (_ca[0]-2) * (_ca[1]-2) ] = ix + _ca[0]*(iy + (_ca[2] - 2)*_ca[1])
                    m += 1
            m += (_ca[0]-2)*(_ca[1]-2)

            for ix in range(1, _ca[0] - 1):
                for iz in range(2, _ca[2] - 2):
                        self._boundary_cells[m] = ix + _ca[0]*(1 + iz*_ca[1])
                        self._boundary_cells[m + (_ca[0]-2) * (_ca[2]-4) ] = ix + _ca[0]*((_ca[1] - 2) + iz*_ca[1])
                        m += 1

            m += (_ca[0]-2)*(_ca[2]-4)

            for iy in range(2, _ca[1] - 2):
                for iz in range(2, _ca[2] - 2):
                        self._boundary_cells[m] = 1 + _ca[0]*(iy + iz*_ca[1])
                        self._boundary_cells[m + (_ca[1]-4) * (_ca[2]-4) ] = _ca[0]-2 + _ca[0]*(iy + iz*_ca[1])
                        m += 1

            m += (_ca[1]-4)*(_ca[2]-4)
            

            self._boundary_cell_version = self._cell_array.version


        return self._boundary_cells

    @property
    def cell_array(self):
        """
        Return cell array.
        """
        return self._cell_array

    @property
    def halos(self):
        return self._halos

    def set_extent(self, new_extent=np.array([1., 1., 1.])):
        """
        Set domain extents
        
        :arg np.array(3,1) new_extent: New extents.
        
        """
        self._extent[0:4] = new_extent
        self._extent_global[0:4] = new_extent

    def set_cell_array_radius(self, rn):
        """
        Create cell structure based on current extent and extended cutoff distance.
        
        :arg double rn:  :math:`r_n = r_c + \delta`
        
        """

        '''Here everything is global'''

        self._cell_array[0] = int(self._extent[0] / rn)
        self._cell_array[1] = int(self._extent[1] / rn)
        self._cell_array[2] = int(self._extent[2] / rn)

        if runtime.VERBOSE.level > 1:
            pio.pprint("Internal cell array:", self._cell_array, ", Extent:",self._extent)

        self._cell_edge_lengths[0] = self._extent[0] / self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1] / self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2] / self._cell_array[2]


        '''Get a local cell array and perform domain decomp'''
        _ca_tuple, _bs = _create_domain_decomp(self.cell_array, self._periods)
        self.cell_array[0] = _ca_tuple[0]
        self.cell_array[1] = _ca_tuple[1]
        self.cell_array[2] = _ca_tuple[2]

        _top = mpi.MPI_HANDLE.top

        '''Calculate local boundary'''
        '''Cumalitive sum up to self_index - 1 '''
        _Cx = 0
        for ix in range(_top[0]):
            _Cx += _bs[0][ix]

        _Cy = 0
        for ix in range(_top[1]):
            _Cy += _bs[1][ix]

        _Cz = 0
        for ix in range(_top[2]):
            _Cz += _bs[2][ix]

        '''Inner boundary (inside halo cells)'''
        self._boundary = [
            -0.5 * self._extent[0] + _Cx * self._cell_edge_lengths[0],
            -0.5 * self._extent[0] + (_Cx + self._cell_array[0]) * self._cell_edge_lengths[0],
            -0.5 * self._extent[1] + _Cy * self._cell_edge_lengths[1],
            -0.5 * self._extent[1] + (_Cy + self._cell_array[1]) * self._cell_edge_lengths[1],
            -0.5 * self._extent[2] + _Cz * self._cell_edge_lengths[2],
            -0.5 * self._extent[2] + (_Cz + self._cell_array[2]) * self._cell_edge_lengths[2]
        ]
        self._boundary = data.ScalarArray(self._boundary, dtype=ctypes.c_double)

        '''Domain outer boundary including halo cells'''

        self._boundary_outer = [
            -0.5 * self._extent[0] + (_Cx - 1) * self._cell_edge_lengths[0],
            -0.5 * self._extent[0] + (_Cx + 1 + self._cell_array[0]) * self._cell_edge_lengths[0],
            -0.5 * self._extent[1] + (_Cy - 1) * self._cell_edge_lengths[1],
            -0.5 * self._extent[1] + (_Cy + 1 + self._cell_array[1]) * self._cell_edge_lengths[1],
            -0.5 * self._extent[2] + (_Cz - 1) * self._cell_edge_lengths[2],
            -0.5 * self._extent[2] + (_Cz + 1 + self._cell_array[2]) * self._cell_edge_lengths[2]]

        self._boundary_outer = data.ScalarArray(self._boundary_outer, dtype=ctypes.c_double)

        '''Get local extent'''
        self._extent[0] = self._cell_edge_lengths[0] * self._cell_array[0]
        self._extent[1] = self._cell_edge_lengths[1] * self._cell_array[1]
        self._extent[2] = self._cell_edge_lengths[2] * self._cell_array[2]

        '''Increment cell array to include halo'''
        self._cell_array[0] += 2
        self._cell_array[1] += 2
        self._cell_array[2] += 2

        '''Outer extent including halos, used?'''
        self._extent_outer = data.ScalarArray(self._extent.dat + np.array([2, 2, 2]) * self._cell_edge_lengths.dat)

        return True

    @property
    def volume(self):
        """
        Return domain volume.
        """
        return self._extent[0] * self._extent[1] * self._extent[2]

    @property
    def boundary(self):
        """
        Return local domain boundary
        """
        return self._boundary

    @property
    def boundary_outer(self):
        """
        Return local domain boundary
        """
        return self._boundary_outer

    @property
    def extent(self):
        """
        Returns list of domain extents including halo regions.
        """
        # return self._extent_outer
        return self._extent_global

    @property
    def extent_internal(self):
        """
        Returns list of domain extents.
        """

        return self._extent


    def get_shift(self):

        _sfd = host.Array(ncomp=26*3, dtype=ctypes.c_double)

        for dx in range(26):
            dir = mpi.recv_modifiers[dx]

            for ix in range(3):

                if mpi.MPI_HANDLE.top[ix] == 0 and \
                   mpi.MPI_HANDLE.periods[ix] == 1 and \
                   dir[ix] == -1:

                    _sfd[dx*3 + ix] = self.extent[ix]

                elif mpi.MPI_HANDLE.top[ix] == mpi.MPI_HANDLE.dims[ix] - 1 and \
                   mpi.MPI_HANDLE.periods[ix] == 1 and \
                   dir[ix] == 1:

                    _sfd[dx*3 + ix] = -1. * self.extent[ix]

                else:
                    _sfd[dx*3 + ix] = 0.0


        return _sfd

    @property
    def cell_edge_lengths(self):
        """
        Return cell edge lengths.
        """
        return self._cell_edge_lengths

    @property
    def cell_count(self):
        """
        Return cell count for domain.
        """
        return self._cell_array[0] * self._cell_array[1] * self._cell_array[2]




def factor(n):
    return [ix for ix in range(1, n / 2 + 1) if not n % ix] + [n]


def pfactor(n):
    lst = []
    l = 2
    while l <= n:
        if n % l == 0:
            n /= l
            lst.append(l)
        else:
            l += 1
    return lst

def _find_domain_decomp(global_cell_array=None, nproc=None):
    """
    find a decomp
    :param global_cell_array:
    :param nproc:
    :return:
    """

    '''Order domain dimension sizes in descending order'''
    _cal = [[0, global_cell_array[0]], [1, global_cell_array[1]], [2, global_cell_array[2]]]
    _cal.sort(key=lambda x: x[1], reverse=True)


    '''Prime factor number of processes'''
    _factors = pfactor(nproc)

    '''Create grid from factorisation'''
    if len(_factors) == 0:
        _NP = [1, 1, 1]
    elif len(_factors) == 1:
        if _factors[0] > _cal[0][1]:
            print "ERROR: Cannot decompose this domain onto this number of " \
                  "processors"
            quit()

        _NP = [_factors[0], 1, 1]

    elif len(_factors) == 2:
         if _factors[0] > _cal[0][1]:
            print "ERROR: Cannot decompose this domain onto this number of " \
                  "processors"
            quit()
         if _factors[1] > _cal[1][1]:
            print "ERROR: Cannot decompose this domain onto this number of " \
                  "processors"
            quit()

         _NP = [_factors[0], _factors[1], 1]
    else:

        _factors.sort(reverse=True)

        if len(_factors)==4:
            _NP = []
            _NP.append(_factors[0])
            _NP.append(_factors[1])
            _NP.append(_factors[2] * _factors[3])

        elif len(_factors)==5:
            _NP = []
            _NP.append(_factors[0])
            _NP.append(_factors[1] * _factors[2])
            _NP.append(_factors[3] * _factors[4])

        else:
            _factors.sort(reverse=True)
            _q = len(_factors) / 3
            #print _q, _factors[0:_q:], _factors[_q:2 * _q:], _factors[2*_q::]
            _NP = []
            _NP.append(reduce(lambda x, y: x * y, _factors[0:_q:]))
            _NP.append(reduce(lambda x, y: x * y, _factors[_q:2 * _q:]))
            _NP.append(reduce(lambda x, y: x * y, _factors[2 * _q::]))


    '''Order processor calculated dimension sizes in descending order'''
    _NP.sort(reverse=True)


    '''Try to match avaible processor dimensions to phyiscal cells'''

    success = True

    _dims = [0, 0, 0]
    for i in range(3):
        ix = _cal[i][0]
        if _cal[i][1] < _NP[i]:
            print "ERROR matching domain to processes, dimension %(DIM)s" \
                  %{'DIM': str(ix)}
            success = False

        _dims[ix] = _NP[i]


    if not success:
        print "Processor grid error, suitable layout search failed." + str(_dims[:]) + str(global_cell_array[:])
        quit()

    return _dims

def _get_cell_distribution(global_cell_array=None, dims=None, top=None):

    # blocks per cell
    _bsc = [int(math.ceil( float(global_cell_array[0]) / float(dims[0]))),
            int(math.ceil( float(global_cell_array[1]) / float(dims[1]))),
            int(math.ceil( float(global_cell_array[2]) / float(dims[2])))]

    # print dims
    # print _bsc

    # Calculate global distribution of cells
    _bs = []
    for ix in range(3):
        _tmp = []

        if (_bsc[ix]*(dims[ix]-1)) < global_cell_array[ix]:

            for iy in range(dims[ix] - 1):
                _tmp.append(int(_bsc[ix]))
            _tmp.append(int(global_cell_array[0] - (dims[ix] - 1) * _bsc[ix]))

        else:

            R = global_cell_array[ix] % dims[ix]
            for iy in range(R):
                _tmp.append(_bsc[ix])
            for iy in range(dims[ix] - R):
                _tmp.append((global_cell_array[ix] - R * _bsc[ix])/ (dims[ix] - R) )

        assert len(_tmp) == dims[ix], "DD size missmatch, dim: " + str(ix) + " " + str(_tmp[:])
        _tsum = 0
        for tx in _tmp:
            _tsum += tx

        assert _tsum == global_cell_array[ix], "DD failure to assign cells, dim: " + str(ix) + " " + str(_tmp[:])

        _bs.append(_tmp)

    if runtime.VERBOSE.level > 1:
        pio.pprint("Cell layout", _bs)

    # Get local cell array
    local_cell_array = (_bs[0][top[0]], _bs[1][top[1]], _bs[2][top[2]])

    return local_cell_array, _bs



def _create_domain_decomp(global_cell_array=None, periods=None):
    """
    Create a local cell array from a global cell array and decompose the domain
    :param global_cell_array:
    :return:
    """

    _dims = _find_domain_decomp(global_cell_array, mpi.MPI_HANDLE.nproc)

     # Create cartesian communicator
    _dims = tuple(_dims)
    mpi.MPI_HANDLE.create_cart(_dims[::-1],
                               (bool(periods[2]), bool(periods[1]), bool(periods[0])),
                               True)


    # Topology has below indexing, last index reverses
    # [z,y,x]
    _top = mpi.MPI_HANDLE.top
    _cell_distro = _get_cell_distribution(global_cell_array, _dims, _top)

    return _cell_distro[0:2:]





class BoundaryTypePeriodic(object):
    """
    Class to hold and perform periodic boundary conditions.

    :arg state_in: State on which to apply periodic boundaries to.
    """

    def __init__(self, state_in=None):
        self.state = state_in

        # Initialise timers
        self.timer_apply = runtime.Timer(runtime.TIMER, 0)
        self.timer_lib_overhead = runtime.Timer(runtime.TIMER, 0)
        self.timer_search = runtime.Timer(runtime.TIMER, 0)
        self.timer_move = runtime.Timer(runtime.TIMER, 0)

        # One proc PBC lib
        self._one_process_pbc_lib = None
        # Escape guard lib
        self._escape_guard_lib = None
        self._escape_count = None
        self._escape_linked_list = None

        self._flag = host.Array(ncomp=1, dtype=ctypes.c_int)

    def set_state(self, state_in=None):
        assert state_in is not None, "BoundaryTypePeriodic error: No state passed."
        self.state = state_in

    def apply(self):
        """
        Enforce the boundary conditions on the held state.
        """

        self.timer_apply.start()

        self._flag.dat[0] = 0

        if mpi.MPI_HANDLE.nproc == 1:
            """
            BC code for one proc. porbably removable when restricting to large parallel systems.
            """

            self.timer_lib_overhead.start()

            if self._one_process_pbc_lib is None:

                _one_proc_pbc_code = '''

                int _F = 0;

                for(int _ix = 0; _ix < _end; _ix++){

                    if (abs_md(P[3*_ix]) >= 0.5*E[0]){
                        const double E0_2 = 0.5*E[0];
                        const double x = P[3*_ix] + E0_2;

                        if (x < 0){
                            P[3*_ix] = (E[0] - fmod(abs_md(x) , E[0])) - E0_2;
                            _F = 1;
                        }
                        else{
                            P[3*_ix] = fmod( x , E[0] ) - E0_2;
                            _F = 1;
                        }
                    }

                    if (abs_md(P[3*_ix+1]) >= 0.5*E[1]){
                        const double E1_2 = 0.5*E[1];
                        const double x = P[3*_ix+1] + E1_2;

                        if (x < 0){
                            P[3*_ix+1] = (E[1] - fmod(abs_md(x) , E[1])) - E1_2;
                            _F = 1;
                        }
                        else{
                            P[3*_ix+1] = fmod( x , E[1] ) - E1_2;
                            _F = 1;
                        }
                    }

                    if (abs_md(P[3*_ix+2]) >= 0.5*E[2]){
                        const double E2_2 = 0.5*E[2];
                        const double x = P[3*_ix+2] + E2_2;

                        if (x < 0){
                            P[3*_ix+2] = (E[2] - fmod(abs_md(x) , E[2])) - E2_2;
                            _F = 1;
                        }
                        else{
                            P[3*_ix+2] = fmod( x , E[2] ) - E2_2;
                            _F = 1;
                        }
                    }

                }

                F[0] = _F;

                '''

                _one_proc_pbc_kernel = kernel.Kernel('_one_proc_pbc_kernel', _one_proc_pbc_code, None,['math.h', 'stdio.h'], static_args={'_end':ctypes.c_int})
                self._one_process_pbc_lib = build.SharedLib(_one_proc_pbc_kernel, {'P': self.state.positions,
                                                                                   'E': self.state.domain.extent,
                                                                                   'F': self._flag})

            self.timer_lib_overhead.pause()

            self.timer_move.start()
            self._one_process_pbc_lib.execute(static_args={'_end': ctypes.c_int(self.state.n)})
            self.timer_move.pause()


        else:
            #print '-' * 14

            # Create lib to find escaping particles.

            self.timer_lib_overhead.start()

            if self._escape_guard_lib is None:
                ''' Create a lookup table between xor map and linear index for direction '''
                self._bin_to_lin = data.ScalarArray(ncomp=57, dtype=ctypes.c_int)
                _lin_to_bin = data.ScalarArray(ncomp=26, dtype=ctypes.c_int)

                '''linear to xor map'''
                _lin_to_bin[0] = 1 ^ 2 ^ 4
                _lin_to_bin[1] = 2 ^ 1
                _lin_to_bin[2] = 32 ^ 2 ^ 1
                _lin_to_bin[3] = 4 ^ 1
                _lin_to_bin[4] = 1
                _lin_to_bin[5] = 32 ^ 1
                _lin_to_bin[6] = 4 ^ 1 ^ 16
                _lin_to_bin[7] = 1 ^ 16
                _lin_to_bin[8] = 32 ^ 16 ^ 1

                _lin_to_bin[9] = 2 ^ 4
                _lin_to_bin[10] = 2
                _lin_to_bin[11] = 32 ^ 2
                _lin_to_bin[12] = 4
                _lin_to_bin[13] = 32
                _lin_to_bin[14] = 4 ^ 16
                _lin_to_bin[15] = 16
                _lin_to_bin[16] = 32 ^ 16

                _lin_to_bin[17] = 8 ^ 2 ^ 4
                _lin_to_bin[18] = 2 ^ 8
                _lin_to_bin[19] = 32 ^ 2 ^ 8
                _lin_to_bin[20] = 4 ^ 8
                _lin_to_bin[21] = 8
                _lin_to_bin[22] = 32 ^ 8
                _lin_to_bin[23] = 4 ^ 8 ^ 16
                _lin_to_bin[24] = 8 ^ 16
                _lin_to_bin[25] = 32 ^ 16 ^ 8

                '''inverse map, probably not ideal'''
                for ix in range(26):
                    self._bin_to_lin[_lin_to_bin[ix]] = ix

                _escape_guard_code = '''

                int ELL_index = 26;

                for(int _ix = 0; _ix < _end; _ix++){
                    int b = 0;

                    //Check x direction
                    if (P[3*_ix] < B[0]){
                        b ^= 4;
                    }else if (P[3*_ix] >= B[1]){
                        b ^= 32;
                    }

                    //printf("P[0]=%f, b=%d, B[0]=%f, B[1]=%f \\n", P[3*_ix], b, B[0], B[1]);

                    //check y direction
                    if (P[3*_ix+1] < B[2]){
                        b ^= 2;
                    }else if (P[3*_ix+1] >= B[3]){
                        b ^= 16;
                    }

                    //check z direction
                    if (P[3*_ix+2] < B[4]){
                        b ^= 1;
                    }else if (P[3*_ix+2] >= B[5]){
                        b ^= 8;
                    }

                    //If b > 0 then particle has escaped through some boundary
                    if (b>0){

                        /*
                        cout << "BC " << _ix << " dir " << BL[b] << " | B0-B5 " 
                        << B[0] << " " << B[1] << " "
                        << B[2] << " " << B[3] << " "
                        << B[4] << " " << B[5]
                        << " | Rxyz: "
                        << P[3*_ix] << ", "
                        << P[3*_ix + 1] << ", "
                        << P[3*_ix + 2]
                        << endl;
                        */

                        EC[BL[b]]++;        //lookup which direction then increment that direction escape count.

                        ELL[ELL_index] = _ix;            //Add current local id to linked list.
                        ELL[ELL_index+1] = ELL[BL[b]];   //Set previous index to be next element.
                        ELL[BL[b]] = ELL_index;          //Set current index in ELL to be the last index.

                        ELL_index += 2;
                    }

                }

                '''

                '''Number of escaping particles in each direction'''
                self._escape_count = host.Array(np.zeros(26), dtype=ctypes.c_int)

                '''Linked list to store the ids of escaping particles in a similar way to the cell list.

                | [0-25 escape directions, index of first in direction] [26-end current id and index of next id, (id, next_index) ]|

                '''
                self._escape_linked_list = host.Array(-1 * np.ones(26 + 2 * self.state.nt), dtype=ctypes.c_int)

                _escape_dat_dict = {'EC': self._escape_count,
                                    'BL': self._bin_to_lin,
                                    'ELL': self._escape_linked_list,
                                    'B': self.state.domain.boundary,
                                    'P': self.state.positions}

                _escape_kernel = kernel.Kernel('find_escaping_particles',
                                               _escape_guard_code,
                                               None,
                                               ['stdio.h'],
                                               static_args={'_end': ctypes.c_int})

                self._escape_guard_lib = build.SharedLib(_escape_kernel,
                                                         _escape_dat_dict)

            self.timer_lib_overhead.pause()

            # reset linked list
            self._escape_linked_list[0:26:] = -1
            self._escape_count[::] = 0


            self.timer_search.start()
            self._escape_guard_lib.execute(static_args={'_end':self.state.n})
            self.timer_search.pause()

            self.timer_move.start()
            self.state.move_to_neighbour(self._escape_linked_list,
                                         self._escape_count,
                                         self.state.domain.get_shift())
            self.timer_move.pause()

        self.timer_apply.pause()

        return self._flag.dat[0]
























