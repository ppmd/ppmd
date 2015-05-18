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
        
        self._cell_array[0] = int(self._extent[0]/rn)
        self._cell_array[1] = int(self._extent[1]/rn)
        self._cell_array[2] = int(self._extent[2]/rn)
        
        
        
        self._cell_edge_lengths[0] = self._extent[0]/self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1]/self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2]/self._cell_array[2]
        
        
        self._cell_count_internal = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        
        self._cell_array[0] += 2
        self._cell_array[1] += 2
        self._cell_array[2] += 2
        
        
        self._cell_count = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        self._extent_outer = data.ScalarArray(self._extent.Dat+2*self._cell_edge_lengths.Dat)
        
        #_tmp = int(self._cell_array[0]/float(self._MPI.Get_size()))
        
        
        
        #_dims=(int(self._cell_array[0]/_tmp),1,1)
        
        _dims = (1,1,1)
        
        _periods = (True, True, True)
        
        
        
        self._COMM = self._MPI.Create_cart(_dims, _periods,True)
        
        self._rank = self._COMM.Get_rank()
        self._nproc = self._COMM.Get_size()          
        
        
        
        
        
        
        
        
        self.halo_init()
        
        return True
        
     
        
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
        self._halos = halo.HaloCartesianSingleProcess(self._MPI, self._rank, self._nproc, self._cell_array, self._extent)
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
