import numpy as np

class Dat():
    """
    Base class to hold floating point properties of particles
    """
    def __init__(self, N1 = 1, N2 = 1, initial_value = None):
        """
        Creates N1*N2 array with given initial value.
        
        :arg N1: (Integer) First dimension.
        :arg N2: (Integer) Second dimension.
        :arg initial_value: (Float) Value to initialise array with, default 0.0.
    
        """
        
        self._N1 = N1
        self._N2 = N2
        
        if (initial_value != None):
            self._Dat = float(initial_value) * np.ones([N1, N2], dtype=float, order='C')
        else:
            self._Dat = np.zeros([N1, N2], dtype=float, order='C')
        
        
    
    def Dat(self):
        """
        Returns entire data array.
        """
        return self._Dat
        
    def __getitem__(self, ix):
        return self._Dat[ix]
        
    def __str__(self):
        return str(self._Dat)




