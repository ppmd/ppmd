import numpy as np
import math
import ctypes
import data
import kernel
import loop
import halo
import build
from mpi4py import MPI


def factor(N):
    return [ix for ix in range(1, N/2 + 1) if not N%ix] + [N]    
    
def pfactor(N):
    lst=[]
    l=2
    while l<=N:
        if N%l==0:
            N/=l
            lst.append(l)
        else:
            l+=1
    return lst   

class BaseDomain(object):
    '''
    Base class for simulation domain, cartesian, 3D. Initialises domain with given extents.
    
    :arg np.array(3,1) extent: [x,y,z] numpy array with extents of simulation domain.
    :arg int cellcount: Number of cells within domain (optional).

    '''

    def __init__(self, extent = np.array([1., 1., 1.]), cell_count = 1):
        
        
        
        self._extent = data.ScalarArray(extent)
        
        self._cell_count = cell_count
        self._cell_array = data.ScalarArray(np.array([1,1,1]), dtype=ctypes.c_int)
        self._cell_edge_lengths = data.ScalarArray(np.array([1.,1.,1.], dtype=ctypes.c_double))
        
        
        
        self._BCloop = None
        
        
    def BCSetup(self, state):
        '''
        Setup loop to apply periodic boundary conditions to input positions.
        
        :arg particle.Dat positions: particle.Dat containing particle positions.
        '''      
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
        
        self._BCcodeDict = {'P':self._BC_state.positions, 'E':self._extent}
        self._BCkernel= kernel.Kernel('BCkernel', self._BCcode, headers=['math.h'])
        self._BCloop = loop.SingleAllParticleLoop(self._BC_state.positions.npart, self._BCkernel,self._BCcodeDict) 
           
        
    def BCexecute(self):
        
        #self.boundary_correct(self._positions)
        assert self._BCloop != None, "Run BCSetup first"
        self._BCloop.execute()
        
    
    
        
    @property  
    def extent(self):
        """
        Returns list of domain extents.
        """
        return self._extent
                
        
        
    def set_extent(self, new_extent = np.array([1., 1., 1.])):
        """
        Set domain extents
        
        :arg np.array(3,1) new_extent: New extents.
        
        """
        self._extent[0:4] = new_extent
        
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
        self._cell_count = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        self._cell_edge_lengths[0] = self._extent[0]/self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1]/self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2]/self._cell_array[2]
        
    @property    
    def volume(self):
        """
        Return domain volume.
        """
        return self._extent[0]*self._extent[1]*self._extent[2]
        
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

        if (int(self._extent[0]/rn) < 3 or int(self._extent[1]/rn) < 3 or int(self._extent[2]/rn) < 3):
            print "WARNING: Less than three cells per coordinate direction. Cell based domain will not be used"
            
            self._cell_array[0] = 1
            self._cell_array[1] = 1
            self._cell_array[2] = 1  
            self._cell_count_recalc()
            return False
            
        else:
            self._cell_array[0] = int(self._extent[0]/rn)
            self._cell_array[1] = int(self._extent[1]/rn)
            self._cell_array[2] = int(self._extent[2]/rn)            
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
        
        
################################################################################################################
# BASE DOMAIN HALO
################################################################################################################

         
class BaseDomainHalo(BaseDomain):

    def __init__(self, extent = np.array([1., 1., 1.]), cell_count = 1):
        
        self._MPI = MPI.COMM_WORLD
        self._DEBUG = True
        
        self._extent = data.ScalarArray(extent)
        self._cell_count = cell_count
        self._cell_array = data.ScalarArray(np.array([1,1,1]), dtype=ctypes.c_int)
        self._cell_edge_lengths = data.ScalarArray(np.array([1.,1.,1.], dtype=ctypes.c_double))
        
        
        
        self._BCloop = None

    def set_cell_array_radius(self, rn):
        """
        Create cell structure based on current extent and extended cutoff distance.
        
        :arg double rn:  :math:`r_n = r_c + \delta`
        
        """
        
        
        '''Here everything is global'''
        
        self._cell_array[0] = int(self._extent[0]/rn)
        self._cell_array[1] = int(self._extent[1]/rn)
        self._cell_array[2] = int(self._extent[2]/rn)
        
        self._cell_edge_lengths[0] = self._extent[0]/self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1]/self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2]/self._cell_array[2]
        
        self._cell_count_internal = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        
        
        '''Get number of processes'''
        _Np = self._MPI.Get_size()
        
        '''Prime factor number of processes'''
        _factors = pfactor(_Np)
        
        '''Create grid from factorisation'''
        if len(_factors)==0:
            _NP=[1,1,1]
        elif len(_factors)==1:
            _NP=[_factors[0],1,1]
        elif len(_factors)==2:
            _NP=[_factors[0],_factors[1],1]
        else:
            _factors.sort(reverse=True)
            _q = len(_factors)/3
            _NP = []
            _NP.append(reduce(lambda x, y: x*y, _factors[0:_q:]))
            _NP.append(reduce(lambda x, y: x*y, _factors[_q:2*_q:]))
            _NP.append(reduce(lambda x, y: x*y, _factors[2*_q::]))
            
        '''Order processor calculated dimension sizes in descending order'''
        _NP.sort(reverse=True)
        
        '''Order domain dimension sizes in descending order'''
        _cal = [[0,self._cell_array[0]], [1,self._cell_array[1]], [2,self._cell_array[2]]]
        _cal.sort(key=lambda x:x[1], reverse=True)
        
        '''Try to match avaible processor dimensions to phyiscal cells'''
        _dims=[0,0,0]
        for i in range(3):
            ix = _cal[i][0]
            _dims[ix] = _NP[i] 
        
        '''Calculate what cell array sizes would be using given processor grid'''
        _bsc = [math.ceil(self._cell_array[0]/float(_dims[0])),
                math.ceil(self._cell_array[1]/float(_dims[1])),
                math.ceil(self._cell_array[2]/float(_dims[2]))]
        
        '''Round down number of processes per direction if excessive'''
        _dims = [
                int(math.ceil(self._cell_array[0]/_bsc[0])),
                int(math.ceil(self._cell_array[1]/_bsc[1])),
                int(math.ceil(self._cell_array[2]/_bsc[2]))
                ]
        
        '''Create cartesian communicator'''
        self._dims = tuple(_dims)
        self._COMM = self._MPI.Create_cart(self._dims[::-1], (True, True, True),True)
        
        '''get rank, nprocs'''
        self._rank = self._COMM.Get_rank()
        self._nproc = self._COMM.Get_size()          
        
        if self._rank == 0:
            print "Processor count", self._nproc,"Processor layout", self._dims
        
        
        '''Topology has below indexing, last index reverses'''
        #[z,y,x]
        self._top = self._COMM.Get_topo()[2][::-1]
        
        #print "rank", self._rank, "top", self._top
        
        
        '''Calculate global distribtion of cells'''
        _bs = []
        for ix in range(3):
            _tmp = []
            for iy in range(_dims[ix]-1):
                _tmp.append(int(_bsc[ix]))
            _tmp.append(int(self._cell_array[0] - (_dims[ix]-1)*_bsc[ix]))
            _bs.append(_tmp)
        
        #print "bs =", _bs
        
        if self._rank == 0:
            print "Cell layout", _bs
            print "Global extent,", self._extent
                
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
        
        
        self._boundary = [
                         -0.5*self._extent[0] + _Cx*self._cell_edge_lengths[0], -0.5*self._extent[0] + (_Cx+self._cell_array[0])*self._cell_edge_lengths[0],
                         -0.5*self._extent[1] + _Cy*self._cell_edge_lengths[1], -0.5*self._extent[1] + (_Cy+self._cell_array[1])*self._cell_edge_lengths[1],
                         -0.5*self._extent[2] + _Cz*self._cell_edge_lengths[2], -0.5*self._extent[2] + (_Cz+self._cell_array[2])*self._cell_edge_lengths[2]
                         ]
        self._boundary = data.ScalarArray(self._boundary, dtype=ctypes.c_double)
        
        self._boundary_outer = [
                         -0.5*self._extent[0]+(_Cx-1)*self._cell_edge_lengths[0], -0.5*self._extent[0]+(_Cx+1+self._cell_array[0])*self._cell_edge_lengths[0],
                         -0.5*self._extent[1]+(_Cy-1)*self._cell_edge_lengths[1], -0.5*self._extent[1]+(_Cy+1+self._cell_array[1])*self._cell_edge_lengths[1],
                         -0.5*self._extent[2]+(_Cz-1)*self._cell_edge_lengths[2], -0.5*self._extent[2]+(_Cz+1+self._cell_array[2])*self._cell_edge_lengths[2]]        
        
        
        self._boundary_outer = data.ScalarArray(self._boundary_outer, dtype=ctypes.c_double)
        
        
        
        
        '''Get local extent'''
        self._extent[0] = self._cell_edge_lengths[0]*self._cell_array[0]
        self._extent[1] = self._cell_edge_lengths[1]*self._cell_array[1]
        self._extent[2] = self._cell_edge_lengths[2]*self._cell_array[2]
        
        '''Increment cell array to include halo'''
        self._cell_array[0] += 2
        self._cell_array[1] += 2
        self._cell_array[2] += 2        
        
        '''Get local cell count'''
        self._cell_count = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        
        '''Outer extent including halos, used?'''
        self._extent_outer = data.ScalarArray(self._extent.Dat+2*self._cell_edge_lengths.Dat)        
        
        '''Init halos'''
        self.halo_init()
        
        return True
        
    @property
    def boundary(self):
        '''
        Return local domain boundary
        '''
        return self._boundary
        
    @property
    def boundary_outer(self):
        '''
        Return local domain boundary
        '''
        return self._boundary_outer 
        
        
    @property  
    def extent(self):
        """
        Returns list of domain extents including halo regions.
        """
        return self._extent_outer      
        
    
    def halo_init(self):
        '''
        Method to initialise halos for local domain.
        '''
        self._halos = halo.HaloCartesianSingleProcess(self._COMM, self._rank, self._top, self._dims, self._cell_array, self._extent)
    
    
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
        '''
        Return internal cell count.
        '''
        return self._cell_count_internal
        
    @property
    def rank(self):
        return self._rank
        
    def barrier(self):
        self._MPI.Barrier()
    
    def BCSetup(self, state):     
        
        self._BC_state = state
        
        self._escaping_ids = data.ScalarArray(ncomp = 2*self._BC_state.NT, dtype = ctypes.c_int)
        #self._escaping_dir = data.ScalarArray(ncomp = self._BC_state.NT, dtype = ctypes.c_int)
        self._escape_count = data.ScalarArray(ncomp = 26, dtype = ctypes.c_int)
        self._escape_count_total = data.ScalarArray(ncomp = 1, dtype = ctypes.c_int)     
        self._escape_internal_index = data.ScalarArray(ncomp = 1, dtype = ctypes.c_int)
        
        self._internal_index = data.ScalarArray(ncomp = 1, dtype = ctypes.c_int)
        
        
        self._bin_to_lin = data.ScalarArray(ncomp = 57, dtype = ctypes.c_int)
        self._lin_to_bin = data.ScalarArray(ncomp = 26, dtype = ctypes.c_int)
        
         
        #linear to xor map
        self._lin_to_bin[0] = 1^2^4
        self._lin_to_bin[1] = 2^1
        self._lin_to_bin[2] = 32^2^1
        self._lin_to_bin[3] = 4^1
        self._lin_to_bin[4] = 1
        self._lin_to_bin[5] = 32^1
        self._lin_to_bin[6] = 4^1^16
        self._lin_to_bin[7] = 1^16
        self._lin_to_bin[8] = 32^16^1
        
        self._lin_to_bin[9] = 2^4
        self._lin_to_bin[10] = 2
        self._lin_to_bin[11] = 32^2
        self._lin_to_bin[12] = 4
        self._lin_to_bin[13] = 32
        self._lin_to_bin[14] = 4^16
        self._lin_to_bin[15] = 16
        self._lin_to_bin[16] = 32^16
        
        self._lin_to_bin[17] = 8^2^4
        self._lin_to_bin[18] = 2^8
        self._lin_to_bin[19] = 32^2^8
        self._lin_to_bin[20] = 4^8
        self._lin_to_bin[21] = 8
        self._lin_to_bin[22] = 32^8
        self._lin_to_bin[23] = 4^8^16
        self._lin_to_bin[24] = 8^16
        self._lin_to_bin[25] = 32^16^8
        
        #inverse map, probably not ideal
        for ix in range(26):
            self._bin_to_lin[self._lin_to_bin[ix]]=ix
        
        
        _escape_guard_code = '''
        
        int b = 0;
        
        if (P[0] < B[0]){
            b ^= 32;
        }else if (P[0] > B[1]){
            b ^= 4;
        }
        
        if (P[1] < B[2]){
            b ^= 16;
        }else if (P[1] > B[3]){
            b ^= 2;
        }        
        
        if (P[2] < B[4]){
            b ^= 1;
        }else if (P[2] > B[5]){
            b ^= 8;
        }        
        
        if (b>0){
            EC[BL[b]]++;
            ECT[0]++;
            EI[EII[0]] = I[0];
            EI[EII[0]+1] = BL[b];
            EII[0]+=2;
            
        }
        
        I[0]++;
        
        '''
        
        _escape_guard_dict = {'P':self._BC_state.positions,
                              'B':self._boundary,
                              'EC':self._escape_count,
                              'BL':self._bin_to_lin,
                              'ECT':self._escape_count_total,
                              'EI':self._escaping_ids,
                              'EII':self._escape_internal_index,
                              'I':self._internal_index
                              }
        
        
        
        _escape_guard_kernel = kernel.Kernel('FindEscapingParticles', _escape_guard_code, headers=['math.h'])
        self._escape_guard_loop = loop.SingleAllParticleLoop(self._BC_state.positions.npart, _escape_guard_kernel, _escape_guard_dict, DEBUG = self._DEBUG)       
        
        self._escape_send_buffer = data.ScalarArray(ncomp = 7*self._BC_state.NT, dtype = ctypes.c_double)
        
        
        _escape_packing_code = '''
        
        int index = 0;
        for (int d = 0; d < 26; d++){
            for(int ix = 0; ix < ECT[0]; ix++){
                
                if (EI[(2*ix)+1] == d){
                    const int id = EI[2*ix];
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    ESB[index]   = P[LINIDX_2D(3,id,0)];
                    ESB[index+1] = P[LINIDX_2D(3,id,1)];
                    ESB[index+2] = P[LINIDX_2D(3,id,2)];
                    ESB[index+3] = V[LINIDX_2D(3,id,0)];
                    ESB[index+4] = V[LINIDX_2D(3,id,1)];
                    ESB[index+5] = V[LINIDX_2D(3,id,2)];                    
                    ESB[index+6] = (double) EGID[id];
                    index += 7;
                }
            }
        }
        
        '''
        
        
        
        _escape_packing_dict={'P':self._BC_state.positions,
                              'V':self._BC_state.velocities,
                              'EGID':self._BC_state.global_ids,
                              'EC':self._escape_count,
                              'EI':self._escaping_ids,
                              'ESB':self._escape_send_buffer,
                              'ECT':self._escape_count_total
                              }        
        
        _pack_escapees_kernel = kernel.Kernel('PackEscapingParticles', _escape_packing_code, headers = ['stdio.h'])
        self._escape_packing_lib = build.SharedLib(_pack_escapees_kernel, _escape_packing_dict, DEBUG = self._DEBUG)
        
        
        
        
        
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
        
        self._BCcodeDict = {'P':self._BC_state.positions, 'E':self._extent}
        self._BCkernel= kernel.Kernel('BCkernel', self._BCcode, headers=['math.h'])
        self._BCloop = loop.SingleAllParticleLoop(self._BC_state.positions.npart, self._BCkernel,self._BCcodeDict, DEBUG = self._DEBUG) 
    
        
        
    
    
    def BCexecute(self):
        
        
        if (self._nproc == 1):
            self._BCloop.execute()
        else:
            
            self._escaping_ids.resize(2*self._BC_state.N)
            #self._escaping_dir.resize(self._BC_state.N)
            
            self._escape_internal_index.zero()
            self._escape_count_total.zero()
            self._escape_count.zero()
        
            self._escape_guard_loop.execute()
            
            
            
            self._escape_packing_lib.execute()
            
            
            
            
            
            

    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
