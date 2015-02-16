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
        self._cell_edge_lengths = np.array([1.,1.,1.],dtype=float)
      
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
        
        periodic = False
        
        C_in = np.zeros(3,dtype=int)
        C_in[0] = (C_in_abs[0]-1) % self._cell_array[0] + 1
        C_in[1] = (C_in_abs[1]-1) % self._cell_array[1] + 1
        C_in[2] = (C_in_abs[2]-1) % self._cell_array[2] + 1
        
        correction = np.array([((C_in[2] - 1)*self._cell_array[1] + C_in[1] - 1)*self._cell_array[0] + C_in[0],0,0,0])
        
        if (C_in[0] != C_in_abs[0] or C_in[1] != C_in_abs[1] or C_in[2] != C_in_abs[2]):
            periodic = True
            correction[1:4:1] = np.sign(C_in_abs-C_in)
            

        return [correction,periodic]
        
        
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
        
    def get_adjacent_cells(self,ix):
        """
        Returns the 14 neighbouring cells as linear index.
        
        :arg ix: (int) Input index.
        
        """
         
        cell_list = np.zeros([14,4],dtype=int)
        cell_list_boundary=[]
        
        C = self.cell_index_tuple(ix)
        
        stencil_map = [
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [1,-1,0],
            [-1,1,1],
            [0,1,1],
            [1,1,1],
            [-1,0,1],
            [0,0,1],
            [1,0,1],
            [-1,-1,1],
            [0,-1,1],
            [1,-1,1]
            ]
        
        for ix in range(14):

            ind = stencil_map[ix]

            tmp=self.cell_index_lin_offset(C+ind)
            cell_list[ix,] = tmp[0]
            cell_list_boundary.append(tmp[1])
        
 
        return [cell_list, cell_list_boundary]
            
        
        
        
        
        
        
        
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
        
        if (self._cell_array[0] < 3 or self._cell_array[1] < 3 or self._cell_array[2] < 3):
            print "WARNING: Less than three cells per coordinate direction."
            print "Cell array = ", self._cell_array
            print "Domain extents = ",self._extent
        
        
        
        self._cell_count_recalc()
        
        
        
        
        
        
        
    def cell_array(self, cell_array):
        """
        Return cell array.
        """
        
        return self._cell_array   
    
    
    
    
    
    
    
    
