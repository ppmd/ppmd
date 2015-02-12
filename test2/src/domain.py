import numpy as np

class BaseDomain():
    '''
    Base class for simulation domain.

    '''

    def __init__(self, extent = np.array([1., 1., 1.]), cellcount = None):
        """
        Initialises a domain with a list length three.
        
        :arg extent: [x,y,z] numpy array with extents of simulation domain.
        :arg cellcount: (Integer) Number of cells within domain.
        
        """
        self._extent = extent
        self._cellcount = cellcount
    
      
    def extent(self):
        """
        Returns list of domain extents.
        """
        return self._extent
        
    def set_extent(self, new_extent = np.array([1., 1., 1.])):
        """
        Set domain extents
        
        :arg: (np.shape(1,3)) New extents.
        
        """
        self._extent = new_extent
        
    
    def cellcount(self):
        """
        Return cell count for domain.
        """
        return self._cellcount
