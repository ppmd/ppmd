class domain():
    def __init__(self, extent):
        """
        Expects list of dimensions
        """
        self._extent = extent
    
    @property  
    def dim(self):
        return self._extent
    
        
