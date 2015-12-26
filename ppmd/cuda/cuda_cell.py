"""
CUDA implementations of methods to handle the cell decomposition of a domain.
"""

#system
import ctypes
import numpy as np

#package


#cuda
import cuda_base
import cuda_build

class CellOccupancyMatrix(object):
    """
    Class to compute and store a cell occupancy matrix for a domain and a set of positions
    """
    def __init__(self):
        self._init = False
        self._setup = False

        self.cell_contents_count = None
        """Number of particles per cell, determines number of layers per cell."""

        self.cell_reverse_lookup = None
        """Map between particle index and containing cell."""

        self.particle_layers = None
        """Stores which layer each particle is contained in."""

        self.matrix = None
        """The occupancy matrix."""

        # build vars
        self._lib = None
        self._boundary = None
        self._cell_edge_lengths = None
        self._cell_array = None

        # setup vars
        self._n_func = None
        self._domain = None
        self._positions = None

    def setup(self, n_func=None, positions_in=None, domain_in=None):
        """
        Setup the cell occupancy matrix class
        :param n_func:
        :param positions_in:
        :param domain_in:
        :return:
        """
        assert n_func is not None, "No n_func passed."
        assert positions_in is not None, "No positions passed"
        assert domain_in is not None, "No domain passed"

        self._n_func = n_func
        self._domain = domain_in
        self._positions = positions_in

        self.particle_layers = cuda_base.Array(ncomp=self._n_func(), dtype=ctypes.c_int)
        self.cell_reverse_lookup = cuda_base.Array(ncomp=self._n_func(), dtype=ctypes.c_int)
        self.cell_contents_count = cuda_base.Array(ncomp=self._domain.cell_count, dtype=ctypes.c_int)
        self.matrix = cuda_base.Matrix(nrow=self._domain.cell_count,
                                       ncol=self._n_func()/self._domain.cell_count,
                                       dtype=ctypes.c_int)

        self._boundary = cuda_base.Array(initial_value=self._domain.boundary_outer)
        self._cell_edge_lengths = cuda_base.Array(initial_value=self._domain.cell_edge_lengths)
        self._cell_array = cuda_base.Array(initial_value=self._domain.cell_array, dtype=ctypes.c_int)

        self._setup = True
        self._build()

    def _build(self):
        """
        Build the library to create the cell occupancy matrix.
        :return:
        """
        assert self._setup is not False, "Run CellOccupancyMatrix.setup() first."

        _p1_header_code = '''
        //Header

        extern "C" int LayerSort();


        '''

        _p1_code = '''
        //source

        int LayerSort(){
            int err = 2;


            return err;
        }
        '''

        _p1_src = cuda_build.source_write(_p1_header_code, _p1_code, 'CellOccupancyMatrix')
        _p1_lib_f = cuda_build.cuda_build_lib(_p1_src[0], hash=False)
        _p1_lib = cuda_build.load(_p1_lib_f)

        print _p1_lib['LayerSort']()



        self._init = True
















