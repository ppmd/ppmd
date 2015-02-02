class BaseDomain():
    '''
    Base class for simulation domain.

    '''

    def __init__(self, extent = [1.0, 1.0, 1.0], cellcount = 1):
        """
        Initialises a domain with a list length three.
        
        :arg extent: [x,y,z] - extents of simulation domain.
        
        """
        self._extent = extent
        self._cellcount = cellcount
    
      
    def extent(self):
        """
        Returns list of domain extents.
        """
        return self._extent
    
    def cellcount(self):
        """
        Return cell count for domain.
        """
        return self._cellcount
