import numpy as np
import math
import ctypes


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
        self._cell_array = np.array([1,1,1], dtype=ctypes.c_int, order='C')
        self._cell_edge_lengths = np.array([1.,1.,1.],dtype=float)
        
        self._USE_C = False
        if (self._USE_C):
            self._periodic_boundary = np.ctypeslib.load_library('libperiodic_boundary.so','.')
            self._periodic_boundary.d_periodic_boundary.restype = ctypes.c_int
            self._periodic_boundary.d_periodic_boundary.argtypes = [ ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)] 
            
        
        
      
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
        
    def cell_index_lin(self,C_in):
        """
        Convert tuple index to linear index. Applies periodic boundaries first.
        
        :arg c_in: Input index to convert.
        """
        
        C_in[0] = (C_in[0]-1) % self._cell_array[0] + 1
        C_in[1] = (C_in[1]-1) % self._cell_array[1] + 1        
        C_in[2] = (C_in[2]-1) % self._cell_array[2] + 1    
        

        return ((C_in[2] - 1)*self._cell_array[1] + C_in[1] - 1)*self._cell_array[0] + C_in[0]
        
    def cell_index_lin_offset(self,C_in_abs):
        """
        Convert tuple index to linear index. Applies periodic boundaries first. Returns periodic boundary offsets.
        
        :arg c_in_abs: Input index to convert.
        """
        
        
        C_in = np.zeros(3,dtype=ctypes.c_int)
        C_in[0] = (C_in_abs[0]-1) % self._cell_array[0] + 1
        C_in[1] = (C_in_abs[1]-1) % self._cell_array[1] + 1
        C_in[2] = (C_in_abs[2]-1) % self._cell_array[2] + 1
        
        correction = np.array([((C_in[2] - 1)*self._cell_array[1] + C_in[1] - 1)*self._cell_array[0] + C_in[0],0,0,0,0],dtype=ctypes.c_int)
        
        if (C_in[0] != C_in_abs[0] or C_in[1] != C_in_abs[1] or C_in[2] != C_in_abs[2]):
            correction[4] = 1
            correction[1:4:1] = np.sign(C_in_abs-C_in)
         

        return correction
        
        
    def cell_index_tuple(self,c_in):
        """
        Convert cell linear index to vector.
        
        :arg c_in: (int) Input index.
        """    
        Cz = 1 + (c_in-1)/(self._cell_array[0]*self._cell_array[1])
        Cx = 1 + (c_in-1) % self._cell_array[0]
        Cy = 1 + int((c_in - (Cz-1)*(self._cell_array[0]*self._cell_array[1]) -1)/(self._cell_array[0]))

        return np.array([Cx,Cy,Cz])
        
        
    def get_cell_lin_index(self,r_in):
        """
        Returns the linear cell index for a given input coordinate.
        
        :arg r_in: (np.array(3,1)) Cartesian vector for particle position.
        """
        
        
        
        r_p = r_in + 0.5*self._extent
        
        
        
        
        Cx = int(r_p[0]/self._cell_edge_lengths[0]) + 1  
        Cy = int(r_p[1]/self._cell_edge_lengths[1]) + 1
        Cz = int(r_p[2]/self._cell_edge_lengths[2]) + 1
        
        
        return ((Cz - 1)*self._cell_array[1] + Cy - 1)*self._cell_array[0] + Cx
        

        
        
        
        
        
        
        
    def _cell_count_recalc(self):
        """    
        Recalculates number of cells in domain. Alongside computing cell edge lengths.
        """
        self._cell_count = self._cell_array[0]*self._cell_array[1]*self._cell_array[2]
        self._cell_edge_lengths[0] = self._extent[0]/self._cell_array[0]
        self._cell_edge_lengths[1] = self._extent[1]/self._cell_array[1]
        self._cell_edge_lengths[2] = self._extent[2]/self._cell_array[2]
        
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
        self._cell_array = cell_array.astype(ctypes.c_int)
        self._cell_count_recalc()
        
        
    def set_cell_array_radius(self, rn):
        """
        Create cell structure based on current extent and extended cutoff distance.
        
        :arg rn: (float) :math:`r_n = r_c + \delta`
        
        """
        self._cell_array[0] = int(self._extent[0]/rn)
        self._cell_array[1] = int(self._extent[1]/rn)
        self._cell_array[2] = int(self._extent[2]/rn)
        
        if (self._cell_array[0] < 3 or self._cell_array[1] < 3 or self._cell_array[2] < 3):
            print "WARNING: Less than three cells per coordinate direction."
            print "Cell array = ", self._cell_array
            print "Domain extents = ",self._extent
        
        
        
        self._cell_count_recalc()
        
        
    
        
    def cell_array(self):
        """
        Return cell array.
        """
        
        return self._cell_array   
    
    
    def boundary_correct(self, input_state):
        """
        Return a new position accounting for periodic boundaries. Would probably benefit from being in C.
        
        :arg r_in: np.array(1,3) input position
        """

        
        
            
        #H = lambda x: 0 if x < 0 else 1
        

        
        
        
        if (self._USE_C != True):
                N = input_state.N()
                r_in = input_state.positions().Dat()
        
        
                for ix in range(N):
                    #if (math.isnan(r_in[ix,0]) or math.isnan(r_in[ix,1]) or math.isnan(r_in[ix,2])):
                    #    print "BC before isnan error", ix, r_in[ix,]      

                    if (abs(r_in[ix,0]) > 0.5*self._extent[0]):
                        x=r_in[ix,0]+0.5*self._extent[0]
                        #r_in[ix,0] = H(x)*( x % self._extent[0] ) + H(-1*x)*(self._extent[0] - (abs(x) % self._extent[0])) - 0.5*self._extent[0]
                        
                        if (x < 0):
                            r_in[ix,0] = (self._extent[0] - (abs(x) % self._extent[0])) - 0.5*self._extent[0]
                        else:
                            r_in[ix,0] = ( x % self._extent[0] ) - 0.5*self._extent[0]
                        
                    if (abs(r_in[ix,1]) > 0.5*self._extent[1]):
                        x=r_in[ix,1]+0.5*self._extent[1]
                        #r_in[ix,1] = H(x)*( x % self._extent[1] ) + H(-1*x)*(self._extent[1] - (abs(x) % self._extent[1])) - 0.5*self._extent[1]        
                        if (x < 0):
                            r_in[ix,1] = (self._extent[1] - (abs(x) % self._extent[1])) - 0.5*self._extent[1]
                        else:
                            r_in[ix,1] = ( x % self._extent[1] ) - 0.5*self._extent[1]

                    if (abs(r_in[ix,2]) > 0.5*self._extent[2]):
                        x=r_in[ix,2]+0.5*self._extent[2]
                        #r_in[ix,2] = H(x)*( x % self._extent[2] ) + H(-1*x)*(self._extent[2] - (abs(x) % self._extent[2])) - 0.5*self._extent[2]        
                        if (x < 0):
                            r_in[ix,2] = (self._extent[2] - (abs(x) % self._extent[2])) - 0.5*self._extent[2]
                        else:
                            r_in[ix,2] = ( x % self._extent[2] ) - 0.5*self._extent[2]
        else:
            args = [self._extent.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    input_state.positions().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double))]
            self._periodic_boundary.d_periodic_boundary(ctypes.c_int(input_state.N()), *args )

        
        
        

    
    
    
    
