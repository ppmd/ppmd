import numpy as np
import math
import ctypes
import data
import kernel
import loop

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
        self._cell_array[0] = int(self._extent[0]/rn)
        self._cell_array[1] = int(self._extent[1]/rn)
        self._cell_array[2] = int(self._extent[2]/rn)
        
        
        if (self._cell_array[0] < 3 or self._cell_array[1] < 3 or self._cell_array[2] < 3):
            print "WARNING: Less than three cells per coordinate direction. Correcting"
            print "Cell array = ", self._cell_array
            print "Domain extents = ",self._extent
        
            self._extent[0] = 3*rn
            self._extent[1] = 3*rn
            self._extent[2] = 3*rn
            self.set_cell_array_radius(rn)
        
        self._cell_count_recalc()
        
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
        
               

    
    
    
    
