import numpy as np
import math

class BaseDomain():
    '''
    Base class for simulation domain, cartesian, 3D.
    
    :arg extent: [x,y,z] numpy array with extents of simulation domain.
    :arg cellcount: (Integer) Number of cells within domain.

    '''

    def __init__(self, extent = np.array([1., 1., 1.]), cell_count = 1):
        """
        Initialises a domain with a list length three.
        
        :arg extent: [x,y,z] numpy array with extents of simulation domain.
        :arg cellcount: (Integer) Number of cells within domain.
        
        """
        self._extent = extent
        self._cell_count = cell_count
        self._cell_array = np.array([1,1,1],dtype=int)
    
      
    def extent(self):
        """
        Returns list of domain extents.
        """
        return self._extent
        
    def set_extent(self, new_extent = np.array([1., 1., 1.])):
        """
        Set domain extents
        
        :arg new_extent: (np.shape(1,3)) New extents.
        
        """
        self._extent = new_extent
        
    
    def cell_count(self):
        """
        Return cell count for domain.
        """
        return self._cell_count
        
    def _cell_count_recalc(self):
        """    
        Recalculates number of cells in domain. 
        """
        self._cell_count = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        
    def volume(self):
        """
        Return domain volume.
        """
        return self._extent[0]*self._extent[1]*self._extent[2]
        
    def set_cell_array_explicit(self, cell_array):
        """
        Set cell array with a vector.
        
        :arg cell_array: (np.array(1,3), new cell array.)
        """
        self._cell_array = cell_array.astype(int)
        self._cell_count_recalc()
        
        
    def set_cell_array_radius(self, rn):
        """
        Create cell structure based on current extent and extended cutoff distance.
        
        :arg rn: (float) :math:`r_n = r_c + \delta`
        
        """
        self._cell_array[0] = int(self._extent[0]/rn)
        self._cell_array[1] = int(self._extent[1]/rn)
        self._cell_array[2] = int(self._extent[2]/rn)
        self._cell_count_recalc()
        
        
        
        
        
        
        
    def cell_array(self, cell_array):
        """
        Return cell array.
        """
        
        return self._cell_array   
    
    
    
    
    
    
    
    
