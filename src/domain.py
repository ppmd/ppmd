import numpy as np
import math
import ctypes
import data
import kernel
import loop
import halo
import build
from mpi4py import MPI
import runtime
import pio

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


class BaseDomain(object):
    """
    Base class for simulation domain, cartesian, 3D. Initialises domain with given extents.
    
    :arg np.array(3,1) extent: [x,y,z] numpy array with extents of simulation domain.
    :arg int cellcount: Number of cells within domain (optional).

    """

    def __init__(self, nt=1, extent=np.array([1., 1., 1.]), cell_count=1):

        self._NT = nt

        self._COMM = None

        self._extent = data.ScalarArray(extent)

        self._cell_count = cell_count
        self._cell_array = data.ScalarArray(np.array([1, 1, 1]), dtype=ctypes.c_int)
        self._cell_edge_lengths = data.ScalarArray(np.array([1., 1., 1.], dtype=ctypes.c_double))

        self._BCloop = None
        self._boundary = [
            -0.5 * self._extent[0],
            0.5 * self._extent[0],
            -0.5 * self._extent[1],
            0.5 * self._extent[1],
            -0.5 * self._extent[2],
            0.5 * self._extent[2]
        ]
        self._boundary = data.ScalarArray(self._boundary, dtype=ctypes.c_double)
        self._halos = False

    @property
    def comm(self):
        return self._COMM

    def bc_setup(self, state):
        """
        Setup loop to apply periodic boundary conditions to input positions.
        
        :arg particle.Dat positions: particle.Dat containing particle positions.
        """
        self._BC_state = state
        self._BCcode = '''
        
        if (abs_md(P[0]) > 0.5*E[0]){
            const double E0_2 = 0.5*E[0];
            const double x = P[0] + E0_2;
            
            if (x < 0){
                P[0] = (E[0] - fmod(abs_md(x) , E[0])) - E0_2;
            }
            else{
                P[0] = fmod( x , E[0] ) - E0_2;
            }
        }
        
        if (abs_md(P[1]) > 0.5*E[1]){
            const double E1_2 = 0.5*E[1];
            const double x = P[1] + E1_2;
            
            if (x < 0){
                P[1] = (E[1] - fmod(abs_md(x) , E[1])) - E1_2;
            }
            else{
                P[1] = fmod( x , E[1] ) - E1_2;
            }
        }
        
        if (abs_md(P[2]) > 0.5*E[2]){
            const double E2_2 = 0.5*E[2];
            const double x = P[2] + E2_2;
            
            if (x < 0){
                P[2] = (E[2] - fmod(abs_md(x) , E[2])) - E2_2;
            }
            else{
                P[2] = fmod( x , E[2] ) - E2_2;
            }
        }                

        '''

        self._BCcodeDict = {'P': self._BC_state.positions, 'E': self._extent}
        self._BCkernel = kernel.Kernel('BCkernel_simple', self._BCcode, headers=['math.h'])
        self._BCloop = loop.SingleAllParticleLoop(self._BC_state.n, self._BC_state.types_map, self._BCkernel, self._BCcodeDict)

    def bc_execute(self):
        # self.boundary_correct(self._positions)
        assert self._BCloop is not None, "Run bc_setup first"

        self._BCloop.execute()

    @property
    def extent(self):
        """
        Returns list of domain extents.
        """
        return self._extent

    def set_extent(self, new_extent=np.array([1., 1., 1.])):
        """
        Set domain extents
        :arg np.array(3,1) new_extent: New extents.
        """
        self._extent[0:4:] = new_extent

        self._boundary = [
            -0.5 * self._extent[0],
            0.5 * self._extent[0],
            -0.5 * self._extent[1],
            0.5 * self._extent[1],
            -0.5 * self._extent[2],
            0.5 * self._extent[2]
        ]
        self._boundary = data.ScalarArray(self._boundary, dtype=ctypes.c_double)

    @property
    def cell_count(self):
        """
        Return cell count for domain.
        """
        return self._cell_count

    def _cell_count_recalc(self):
        """    
        Recalculates number of cells in domain. Alongside computing cell edge lengths.
        """
        self._cell_count = self._cell_array[0] * self._cell_array[1] * self._cell_array[2]
        self._cell_edge_lengths[0] = self._extent[0] / self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1] / self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2] / self._cell_array[2]

    @property
    def volume(self):
        """
        Return domain volume.
        """
        return self._extent[0] * self._extent[1] * self._extent[2]

    def set_cell_array_explicit(self, cell_array):
        """
        Set cell array with a vector.
        
        :arg np.array(3,1) cell_array: new cell array.
        """

        self._cell_array[0:4] = cell_array
        self._cell_count_recalc()

    def set_cell_array_radius(self, rn):
        """
        Create cell structure based on current extent and extended cutoff distance.
        
        :arg double rn:  :math:`r_n = r_c + \delta`
        """

        if (int(self._extent[0] / rn) < 3) or (int(self._extent[1] / rn) < 3) or (int(self._extent[2] / rn) < 3):
            print "WARNING: Less than three cells per coordinate direction. Cell based domain will not be used"

            self._cell_array[0] = 1
            self._cell_array[1] = 1
            self._cell_array[2] = 1
            self._cell_count_recalc()
            return False

        else:
            self._cell_array[0] = int(self._extent[0] / rn)
            self._cell_array[1] = int(self._extent[1] / rn)
            self._cell_array[2] = int(self._extent[2] / rn)
            self._cell_count_recalc()

        return True

    @property
    def cell_array(self):
        """
        Return cell array.
        """

        return self._cell_array

    @property
    def cell_edge_lengths(self):
        """
        Return cell edge lengths.
        """
        return self._cell_edge_lengths

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
        return self._boundary

    @property
    def halos(self):
        return self._halos

##############################################################################################################
# BASE DOMAIN HALO
##############################################################################################################


class BaseDomainHalo(BaseDomain):
    def __init__(self, nt=1, extent=np.array([1., 1., 1.]), cell_count=1, periods=(1, 1, 1)):

        self._NT = nt

        self._periods = periods

        self._MPI_handle = runtime.MPI_HANDLE
        self._MPI_handle.set_periods = self._periods
        self._MPI = MPI.COMM_WORLD
        self._MPIstatus = MPI.Status()
        self._DEBUG = True

        self._extent = data.ScalarArray(extent)

        self._extent_global = data.ScalarArray(extent)

        self._cell_count = cell_count
        self._cell_array = data.ScalarArray(np.array([1, 1, 1]), dtype=ctypes.c_int)
        self._cell_edge_lengths = data.ScalarArray(np.array([1., 1., 1.], dtype=ctypes.c_double))

        self._BCloop = None
        self._halos = False

    @property
    def mpi_handle(self):
        return self._MPI_handle

    @mpi_handle.setter
    def mpi_handle(self, handle):
        self._MPI_handle = handle

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
            pio.pprint("Global cell array:", self._cell_array, ", Global cell extent:",self._extent)

        self._cell_edge_lengths[0] = self._extent[0] / self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1] / self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2] / self._cell_array[2]

        self._cell_count_internal = self._cell_array[0] * self._cell_array[1] * self._cell_array[2]

        '''Get number of processes'''
        _Np = self._MPI.Get_size()

        '''Prime factor number of processes'''
        _factors = pfactor(_Np)

        '''Create grid from factorisation'''
        if len(_factors) == 0:
            _NP = [1, 1, 1]
        elif len(_factors) == 1:
            _NP = [_factors[0], 1, 1]
        elif len(_factors) == 2:
            _NP = [_factors[0], _factors[1], 1]
        else:
            _factors.sort(reverse=True)
            _q = len(_factors) / 3
            _NP = []
            _NP.append(reduce(lambda x, y: x * y, _factors[0:_q:]))
            _NP.append(reduce(lambda x, y: x * y, _factors[_q:2 * _q:]))
            _NP.append(reduce(lambda x, y: x * y, _factors[2 * _q::]))

        '''Order processor calculated dimension sizes in descending order'''
        _NP.sort(reverse=True)

        '''Order domain dimension sizes in descending order'''
        _cal = [[0, self._cell_array[0]], [1, self._cell_array[1]], [2, self._cell_array[2]]]
        _cal.sort(key=lambda x: x[1], reverse=True)

        '''Try to match avaible processor dimensions to phyiscal cells'''
        _dims = [0, 0, 0]
        for i in range(3):
            ix = _cal[i][0]
            _dims[ix] = _NP[i]

        '''Calculate what cell array sizes would be using given processor grid'''
        _bsc = [math.ceil(self._cell_array[0] / float(_dims[0])),
                math.ceil(self._cell_array[1] / float(_dims[1])),
                math.ceil(self._cell_array[2] / float(_dims[2]))]

        '''Round down number of processes per direction if excessive'''
        _dims = [
            int(math.ceil(self._cell_array[0] / _bsc[0])),
            int(math.ceil(self._cell_array[1] / _bsc[1])),
            int(math.ceil(self._cell_array[2] / _bsc[2]))
        ]

        '''Create cartesian communicator'''
        self._dims = tuple(_dims)
        self._COMM = self._MPI.Create_cart(self._dims[::-1],
                                           (bool(self._periods[2]), bool(self._periods[1]), bool(self._periods[0])),
                                           True)

        '''Set the simulation mpi handle to be the newly created one'''
        if self._MPI_handle is not None:
            self._MPI_handle.comm = self._COMM

        '''get rank, nprocs'''
        self._rank = self._COMM.Get_rank()
        self._nproc = self._COMM.Get_size()

        if runtime.VERBOSE.level > 1:
            pio.pprint("Processor count ", self._nproc, " Processor layout ", self._dims)

        '''Topology has below indexing, last index reverses'''
        # [z,y,x]
        self._top = self._COMM.Get_topo()[2][::-1]

        '''Calculate global distribtion of cells'''
        _bs = []
        for ix in range(3):
            _tmp = []
            for iy in range(_dims[ix] - 1):
                _tmp.append(int(_bsc[ix]))
            _tmp.append(int(self._cell_array[0] - (_dims[ix] - 1) * _bsc[ix]))
            _bs.append(_tmp)

        # print "bs =", _bs

        if runtime.VERBOSE.level > 1:
            pio.pprint("Cell layout", _bs)

        '''Get local cell array'''
        self._cell_array[0] = _bs[0][self._top[0]]
        self._cell_array[1] = _bs[1][self._top[1]]
        self._cell_array[2] = _bs[2][self._top[2]]

        '''Calculate local boundary'''
        '''Cumalitive sum up to self_index - 1 '''
        _Cx = 0
        for ix in range(self._top[0]):
            _Cx += _bs[0][ix]

        _Cy = 0
        for ix in range(self._top[1]):
            _Cy += _bs[1][ix]

        _Cz = 0
        for ix in range(self._top[2]):
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

        '''Get local cell count'''
        self._cell_count = self._cell_array[0] * self._cell_array[1] * self._cell_array[2]

        '''Outer extent including halos, used?'''
        self._extent_outer = data.ScalarArray(self._extent.dat + np.array([2, 2, 2]) * self._cell_edge_lengths.dat)

        '''Init halos'''
        self.halo_init()

        return True

    @property
    def mpi_handle(self):
        return self._MPI_handle

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

    def halo_init(self):
        """
        Method to initialise halos for local domain.
        """
        self._halos = halo.HaloCartesianSingleProcess(self._NT, self._MPI_handle, self._cell_array, self._extent_global)

    @property
    def halos(self):
        return self._halos

    @property
    def extent_internal(self):
        """
        Returns list of domain extents.
        """

        return self._extent

    @property
    def cell_count_internal(self):
        """
        Return internal cell count.
        """
        return self._cell_count_internal

    @property
    def rank(self):
        return self._rank

    def barrier(self):
        self._MPI.Barrier()

    def bc_setup(self, state):

        self._BC_state = state

        '''Array to store the local id of scaling particles'''
        self._escaping_ids = data.ScalarArray(ncomp=2 * self._BC_state.nt(), dtype=ctypes.c_int)

        '''Number of escaping particles in each direction'''
        self._escape_count = data.ScalarArray(ncomp=26, dtype=ctypes.c_int)

        '''Total number of escapees'''
        self._escape_count_total = data.ScalarArray(ncomp=1, dtype=ctypes.c_int)

        '''Temporary indices for library'''
        self._escape_internal_index = data.ScalarArray(ncomp=1, dtype=ctypes.c_int)
        self._internal_index = data.ScalarArray(ncomp=1, dtype=ctypes.c_int)

        '''Create a lookup table between xor map and linear index for direction'''
        self._bin_to_lin = data.ScalarArray(ncomp=57, dtype=ctypes.c_int)
        self._lin_to_bin = data.ScalarArray(ncomp=26, dtype=ctypes.c_int)

        '''linear to xor map'''
        self._lin_to_bin[0] = 1 ^ 2 ^ 4
        self._lin_to_bin[1] = 2 ^ 1
        self._lin_to_bin[2] = 32 ^ 2 ^ 1
        self._lin_to_bin[3] = 4 ^ 1
        self._lin_to_bin[4] = 1
        self._lin_to_bin[5] = 32 ^ 1
        self._lin_to_bin[6] = 4 ^ 1 ^ 16
        self._lin_to_bin[7] = 1 ^ 16
        self._lin_to_bin[8] = 32 ^ 16 ^ 1

        self._lin_to_bin[9] = 2 ^ 4
        self._lin_to_bin[10] = 2
        self._lin_to_bin[11] = 32 ^ 2
        self._lin_to_bin[12] = 4
        self._lin_to_bin[13] = 32
        self._lin_to_bin[14] = 4 ^ 16
        self._lin_to_bin[15] = 16
        self._lin_to_bin[16] = 32 ^ 16

        self._lin_to_bin[17] = 8 ^ 2 ^ 4
        self._lin_to_bin[18] = 2 ^ 8
        self._lin_to_bin[19] = 32 ^ 2 ^ 8
        self._lin_to_bin[20] = 4 ^ 8
        self._lin_to_bin[21] = 8
        self._lin_to_bin[22] = 32 ^ 8
        self._lin_to_bin[23] = 4 ^ 8 ^ 16
        self._lin_to_bin[24] = 8 ^ 16
        self._lin_to_bin[25] = 32 ^ 16 ^ 8

        '''inverse map, probably not ideal'''
        for ix in range(26):
            self._bin_to_lin[self._lin_to_bin[ix]] = ix

        '''
        Below code uses the following map between directions and a 6 bit integer.
        
        xu yu zu xl yl zl
        '''

        _escape_guard_code = '''
        
        int b = 0;
        
        //Check x direction
        if (P[0] < B[0]){
            b ^= 32;
        }else if (P[0] > B[1]){
            b ^= 4;
        }
        
        //check y direction
        if (P[1] < B[2]){
            b ^= 16;
        }else if (P[1] > B[3]){
            b ^= 2;
        }        
        
        //check z direction
        if (P[2] < B[4]){
            b ^= 1;
        }else if (P[2] > B[5]){
            b ^= 8;
        }        
        
        //If b > 0 then particle has escaped through some boundary
        if (b>0){
            EC[BL[b]]++;        //lookup which direction then increment that direction escape count.
            ECT[0]++;           //Increment total escape count by 1.
            EI[EII[0]] = I[0];  //In escape ids we have pairs of local index and escape index, here write local index
            EI[EII[0]+1] = BL[b]; //here write escape direction
            EII[0]+=2;
        }
        
        I[0]++;
        
        '''

        _escape_guard_dict = {'P': self._BC_state.positions,
                              'B': self._boundary,
                              'EC': self._escape_count,
                              'BL': self._bin_to_lin,
                              'ECT': self._escape_count_total,
                              'EI': self._escaping_ids,
                              'EII': self._escape_internal_index,
                              'I': self._internal_index
                              }

        _escape_guard_kernel = kernel.Kernel('FindEscapingParticles', _escape_guard_code, headers=['math.h'])
        self._escape_guard_loop = loop.SingleAllParticleLoop(self._BC_state.n, self._BC_state.types_map,
                                                             _escape_guard_kernel, _escape_guard_dict)

        '''Calculate shifts that should be applied when passing though the local domain extents
        Xl 0, Xu 1
        Yl 2, Yu 3
        Zl 4, Zu 5
        '''

        _sf = range(6)
        for ix in range(3):
            if self._top[ix] == 0:
                _sf[2 * ix] = self._extent_global[ix]
            else:
                _sf[2 * ix] = 0.
            if self._top[ix] == self._dims[ix] - 1:
                _sf[2 * ix + 1] = -1. * self._extent_global[ix]
            else:
                _sf[2 * ix + 1] = 0.

        _sfd = [
            _sf[1], _sf[3], _sf[4],  # 0
            0., _sf[3], _sf[4],  # 1
            _sf[0], _sf[3], _sf[4],  # 2
            _sf[1], 0., _sf[4],  # 3
            0., 0., _sf[4],  # 4
            _sf[0], 0., _sf[4],  # 5
            _sf[1], _sf[2], _sf[4],  # 6
            0., _sf[2], _sf[4],  # 7
            _sf[0], _sf[2], _sf[4],  # 8

            _sf[1], _sf[3], 0.,  # 9
            0., _sf[3], 0.,  # 10
            _sf[0], _sf[3], 0.,  # 11
            _sf[1], 0., 0.,  # 12
            _sf[0], 0., 0.,  # 13
            _sf[1], _sf[2], 0.,  # 14
            0., _sf[2], 0.,  # 15
            _sf[0], _sf[2], 0.,  # 16

            _sf[1], _sf[3], _sf[5],  # 17
            0., _sf[3], _sf[5],  # 18
            _sf[0], _sf[3], _sf[5],  # 19
            _sf[1], 0., _sf[5],  # 20
            0., 0., _sf[5],  # 21
            _sf[0], 0., _sf[5],  # 22
            _sf[1], _sf[2], _sf[5],  # 23
            0., _sf[2], _sf[5],  # 24
            _sf[0], _sf[2], _sf[5]  # 25

        ]

        # print self._rank, "LOCAL SHIFTS", _sf


        self._sfd = data.ScalarArray(initial_value=_sfd)

        '''Number of elements to pack'''
        self._ncomp = data.ScalarArray(initial_value=[8], dtype=ctypes.c_int)

        self._escape_send_buffer = data.ScalarArray(ncomp=self._ncomp[0] * self._BC_state.nt(), dtype=ctypes.c_double)

        '''Starting ixdex for packing'''
        self._escape_send_buffer_index = data.ScalarArray(ncomp=26, dtype=ctypes.c_int)

        '''Packing code, needs to become generated based on types that are dynamic and need sending'''

        _escape_packing_code = '''
        
        
        
        ESBi[0] = 0;
        for (int d = 1; d < 26; d++){
            int ei = 0;
            for (int j = 0; j < d; j++){
                ei+= NCOMP[0]*EC[j];
            }
            ESBi[d]=ei;
        }
        
        for (int ix = 0; ix < ECT[0]; ix++){
            
            int id = EI[(2*ix)];
            int d = EI[(2*ix)+1];
            int index = ESBi[d];
            ESBi[d]+= NCOMP[0];
            
            ESB[index]   = P[LINIDX_2D(3,id,0)] + SFD[3*d];
            ESB[index+1] = P[LINIDX_2D(3,id,1)] + SFD[3*d+1];
            ESB[index+2] = P[LINIDX_2D(3,id,2)] + SFD[3*d+2];
            ESB[index+3] = V[LINIDX_2D(3,id,0)];
            ESB[index+4] = V[LINIDX_2D(3,id,1)];
            ESB[index+5] = V[LINIDX_2D(3,id,2)];
            ESB[index+6] = (double) EGID[id];
            ESB[index+7] = (double) TYPE[id];
            
        }
        
        '''
        self._escape_count_recv = data.ScalarArray(ncomp=26, dtype=ctypes.c_int)
        self._escape_recv_buffer = data.ScalarArray(ncomp=self._ncomp[0] * self._BC_state.nt(), dtype=ctypes.c_double)

        _escape_packing_dict = {'P': self._BC_state.positions,
                                'V': self._BC_state.velocities,
                                'EGID': self._BC_state.global_ids,
                                'TYPE': self._BC_state.types,
                                'EC': self._escape_count,
                                'EI': self._escaping_ids,
                                'ESB': self._escape_send_buffer,
                                'ESBi': self._escape_send_buffer_index,
                                'ECT': self._escape_count_total,
                                'SFD': self._sfd,
                                'NCOMP': self._ncomp
                                }

        _pack_escapees_kernel = kernel.Kernel('PackEscapingParticles', _escape_packing_code, headers=['stdio.h'])
        self._escape_packing_lib = build.SharedLib(_pack_escapees_kernel, _escape_packing_dict)

        _recv_modifiers = [
            [-1, -1, -1],  # 0
            [0, -1, -1],  # 1
            [1, -1, -1],  # 2
            [-1, 0, -1],  # 3
            [0, 0, -1],  # 4
            [1, 0, -1],  # 5
            [-1, 1, -1],  # 6
            [0, 1, -1],  # 7
            [1, 1, -1],  # 8

            [-1, -1, 0],  # 9
            [0, -1, 0],  # 10
            [1, -1, 0],  # 11
            [-1, 0, 0],  # 12
            [1, 0, 0],  # 13
            [-1, 1, 0],  # 14
            [0, 1, 0],  # 15
            [1, 1, 0],  # 16

            [-1, -1, 1],  # 17
            [0, -1, 1],  # 18
            [1, -1, 1],  # 19
            [-1, 0, 1],  # 20
            [0, 0, 1],  # 21
            [1, 0, 1],  # 22
            [-1, 1, 1],  # 23
            [0, 1, 1],  # 24
            [1, 1, 1],  # 25
        ]

        self._send_list = [
            ((self._top[0] - ix[0]) % self._dims[0]) + ((self._top[1] - ix[1]) % self._dims[1]) * self._dims[0] + (
                (self._top[2] - ix[2]) % self._dims[2]) * self._dims[0] * self._dims[1] for ix in _recv_modifiers]
        self._recv_list = [
            ((self._top[0] + ix[0]) % self._dims[0]) + ((self._top[1] + ix[1]) % self._dims[1]) * self._dims[0] + (
                (self._top[2] + ix[2]) % self._dims[2]) * self._dims[0] * self._dims[1] for ix in _recv_modifiers]

        self._tmp_index = data.ScalarArray(ncomp=1, dtype=ctypes.c_int)

        _unpacking_code = '''
        
        //printf("before I[0] = %d, ECT = %d, TI[0] = %d |", I[0], ECT[0], TI[0]);
        for (int ix = 0; ix < TI[0]; ix++){
            
            //printf("ix = %d |", ix);
            
            int IX;
            //fill in spaces
            if (ix < ECT[0]) {
                IX = EI[2*ix];
            }
            else {
                //put at end if spaces full
                IX = I[0] + ix - ECT[0];
            }
            
            
            P[LINIDX_2D(3,IX,0)] = ERB[NCOMP[0]*ix];
            P[LINIDX_2D(3,IX,1)] = ERB[NCOMP[0]*ix+1];
            P[LINIDX_2D(3,IX,2)] = ERB[NCOMP[0]*ix+2];
            
            V[LINIDX_2D(3,IX,0)] = ERB[NCOMP[0]*ix+3];
            V[LINIDX_2D(3,IX,1)] = ERB[NCOMP[0]*ix+4];
            V[LINIDX_2D(3,IX,2)] = ERB[NCOMP[0]*ix+5];
            
            RGID[IX] = (int) ERB[NCOMP[0]*ix+6];
            TYPE[IX] = (int) ERB[NCOMP[0]*ix+7];
            
        }
        
        
        //if more were sent than recv'd then we have holes.
        if (TI[0] < ECT[0]){
            
            int ect = ECT[0] - 1;
            
            //int ect_ti = ECT[0] - TI[0] - 2;
            int ect_ti = TI[0];
            
            int eix = -1;
            
            
            while ( (ect_ti < ECT[0]) && (EI[2*ect_ti] < I[0]) ){
                
                eix = EI[2*ect_ti];

                //printf("ect_ti = %d, eix = %d|", ect_ti, eix);
                
                int ti = -1;

                //loop from end to empty slot
                for (int iy = I[0] - 1; iy > eix; iy--){
                    
                    
                    //printf("EI[2*ect] = %d |", EI[2*ect]);
                    
                    if (iy == EI[2*ect]){
                        I[0] = iy;
                        ect--;
                    } else {
                        ti = iy;
                        break;
                    }
                    
                }
                
                //printf("ti = %d |", ti);
                
                
                if (ti > 0){
                    //copy code here from index ti to index eix
                    
                    P[LINIDX_2D(3,eix,0)] = P[LINIDX_2D(3,ti,0)];
                    P[LINIDX_2D(3,eix,1)] = P[LINIDX_2D(3,ti,1)];
                    P[LINIDX_2D(3,eix,2)] = P[LINIDX_2D(3,ti,2)];
                    
                    V[LINIDX_2D(3,eix,0)] = V[LINIDX_2D(3,ti,0)];
                    V[LINIDX_2D(3,eix,1)] = V[LINIDX_2D(3,ti,1)];
                    V[LINIDX_2D(3,eix,2)] = V[LINIDX_2D(3,ti,2)];
                    
                    RGID[eix] = RGID[ti];                  
                    TYPE[eix] = TYPE[ti];
                                        
                    I[0] = ti;
                    
                    
                } else {
                    I[0] = eix;
                    break;  
                }
            
                
                ect_ti++;
                //printf("I[0] = %d |", I[0]);   
            }
            
             
        } else {
        
        I[0] += TI[0] - ECT[0];
        
        }
        
        //printf("after I[0] = %d |", I[0]);
        
        '''

        _unpacking_dict = {'P': self._BC_state.positions,
                           'V': self._BC_state.velocities,
                           'RGID': self._BC_state.global_ids,
                           'ERB': self._escape_recv_buffer,
                           'ECT': self._escape_count_total,
                           'I': self._internal_index,
                           'TI': self._tmp_index,
                           'EI': self._escaping_ids,
                           'NCOMP': self._ncomp,
                           'TYPE': self._BC_state.types
                           }

        _unpacking_kernel = kernel.Kernel('unpackingParticles', _unpacking_code, headers=['stdio.h'])
        self._unpacking_lib = build.SharedLib(_unpacking_kernel, _unpacking_dict)

        self._BCcode = '''
        
        if (abs_md(P[0]) > 0.5*E[0]){
            const double E0_2 = 0.5*E[0];
            const double x = P[0] + E0_2;
            
            if (x < 0){
                P[0] = (E[0] - fmod(abs_md(x) , E[0])) - E0_2;
            }
            else{
                P[0] = fmod( x , E[0] ) - E0_2;
            }
        }
        
        if (abs_md(P[1]) > 0.5*E[1]){
            const double E1_2 = 0.5*E[1];
            const double x = P[1] + E1_2;
            
            if (x < 0){
                P[1] = (E[1] - fmod(abs_md(x) , E[1])) - E1_2;
            }
            else{
                P[1] = fmod( x , E[1] ) - E1_2;
            }
        }
        
        if (abs_md(P[2]) > 0.5*E[2]){
            const double E2_2 = 0.5*E[2];
            const double x = P[2] + E2_2;
            
            if (x < 0){
                P[2] = (E[2] - fmod(abs_md(x) , E[2])) - E2_2;
            }
            else{
                P[2] = fmod( x , E[2] ) - E2_2;
            }
        }                
        
        
        '''

        self._BCcodeDict = {'P': self._BC_state.positions, 'E': self._extent}
        self._BCkernel = kernel.Kernel('BCkernel', self._BCcode, headers=['math.h'])
        self._BCloop = loop.SingleAllParticleLoop(self._BC_state.n, self._BC_state.types_map, self._BCkernel,
                                                  self._BCcodeDict)

    def bc_execute(self):

        if self._nproc == 1:
            self._BCloop.execute()
            # print "normal BCs applied"
        else:

            '''Potentially all could escape'''
            # self._escaping_ids.resize(2*self._BC_state.n())
            # self._escaping_ids.zero()

            '''Zero counts/indices'''
            self._escape_internal_index.zero()
            self._internal_index.zero()

            self._escape_count_total.zero()
            self._escape_count.zero()

            '''Find escaping particles'''
            self._escape_guard_loop.execute()

            '''Exchange sizes'''
            for ix in range(26):
                # print "R", self._rank, "Sending", self._escape_count.Dat[ix:ix+1:], "ix", ix

                self._COMM.Sendrecv(self._escape_count.dat[ix:ix + 1:],
                                    self._send_list[ix],
                                    self._send_list[ix],
                                    self._escape_count_recv.dat[ix:ix + 1:],
                                    self._recv_list[ix],
                                    self._rank,
                                    self._MPIstatus)

                # print "R", self._rank, "RECVD" ,self._escape_count_recv.Dat[ix:ix+1:], "ix", ix

            '''Check packing buffer is large enough then pack'''
            self._escape_send_buffer.resize(self._ncomp[0] * self._escape_count_total[0])
            self._escape_packing_lib.execute()

            '''Exchange packed particle buffers'''
            _sum_send = 0
            _sum_recv = 0

            for ix in range(26):
                self._COMM.Sendrecv(
                    self._escape_send_buffer.dat[_sum_send:_sum_send + self._ncomp[0] * self._escape_count[ix]:],
                    self._send_list[ix],
                    self._send_list[ix],
                    self._escape_recv_buffer.dat[_sum_recv:_sum_recv + self._ncomp[0] * self._escape_count_recv[ix]:],
                    self._recv_list[ix],
                    self._rank,
                    self._MPIstatus)

                _sum_send += self._ncomp[0] * self._escape_count[ix]
                _sum_recv += self._ncomp[0] * self._escape_count_recv[ix]

            self._tmp_index[0] = _sum_recv / self._ncomp[0]

            '''Unpack new particles and compress particle dats'''

            self._BC_state.positions.halo_start_reset()
            self._BC_state.velocities.halo_start_reset()
            self._internal_index[0] = self._BC_state.n()

            self._unpacking_lib.execute()

            self._BC_state.set_n(self._internal_index[0])

            # print "setting halos", self._internal_index[0]

            self._BC_state.positions.halo_start_set(self._internal_index[0])
            self._BC_state.velocities.halo_start_set(self._internal_index[0])
            self._BC_state.forces.halo_start_set(self._internal_index[0])
