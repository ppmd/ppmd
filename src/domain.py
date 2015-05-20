import numpy as np
import math
import ctypes
import data
import kernel
import loop
import halo
from mpi4py import MPI

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
        
        
    def BCSetup(self, positions):
        '''
        Setup loop to apply periodic boundary conditions to input positions.
        
        :arg particle.Dat positions: particle.Dat containing particle positions.
        '''      
        
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
        
        self._BCcodeDict = {'P':positions, 'E':self._extent}
        
        self._BCkernel= kernel.Kernel('BCkernel', self._BCcode, headers=['math.h'])
        
        self._BCloop = loop.SingleAllParticleLoop(positions.npart, self._BCkernel,self._BCcodeDict)       
        #self._positions = positions
           
        
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
        
    def Barrier(self):
        self._COMM.Barrier()
    
    
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
