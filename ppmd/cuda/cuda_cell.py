"""
CUDA implementations of methods to handle the cell decomposition of a domain.
"""

#system
import ctypes
import numpy as np
import math

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

        p1_args = '''const int blocksize[3],
                     const int threadsize[3],
                     const int n,
                     int* __restrict__ d_pl,
                     int* __restrict__ d_crl,
                     int* __restrict__ d_ccc,
                     int* __restrict__ d_M,
                     const int* __restrict__ d_ca,
                     const double* __restrict__ d_b,
                     const double* __restrict__ d_cel,
                     const double* __restrict__ d_p
                     '''

        _p1_header_code = '''
        //Header
        #include <cuda_generic.h>

        extern "C" int LayerSort(%(ARGS)s);


        ''' %{'ARGS': p1_args}

        _p1_code = '''
        //source

        __constant__ int d_n;

        __global__ void d_LayerSort(int* __restrict__ d_pl,
                                    int* __restrict__ d_crl,
                                    int* __restrict__ d_ccc,
                                    const int* __restrict__ d_ca,
                                    const double* __restrict__ d_b,
                                    const double* __restrict__ d_cel,
                                    const double* __restrict__ d_p
        ){

        const int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (_ix < d_n){

            const int C0 = (int)(( d_p[_ix*3]    - d_b[0] ) / d_cel[0]);
            const int C1 = (int)(( d_p[_ix*3]+1  - d_b[2] ) / d_cel[1]);
            const int C2 = (int)(( d_p[_ix*3]+2  - d_b[4] ) / d_cel[2]);

            const int val = (C2*d_ca[1] + C1)*d_ca[0] + C0;

            d_crl[_ix] = val;
            //old=atomicAdd(address, new);
            d_pl[_ix] = atomicAdd(&d_ccc[val], (int)1);


        }
        return;
        }

        int LayerSort(%(ARGS)s){
            int err = 1;
            checkCudaErrors(cudaMemcpyToSymbol(d_n, &n, sizeof(n)));

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];


            d_LayerSort<<<bs,ts>>>(d_pl, d_crl, d_ccc, d_ca, d_b, d_cel, d_p);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" d_LayerSort Execution failed. \\n");

            return err;
        }
        ''' % {'ARGS':p1_args}

        _p1_src = cuda_build.source_write(_p1_header_code, _p1_code, 'CellOccupancyMatrix')
        _p1_lib_f = cuda_build.cuda_build_lib(_p1_src[0], hash=False)
        self._p1_lib = cuda_build.load(_p1_lib_f)




        self._init = True

    def sort(self):

        self.cell_contents_count.zero()

        _tpb = 512
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(self._n_func() / float(_tpb))), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)


        args = [_blocksize,
                _threadsize,
                ctypes.c_int(self._n_func()),
                self.particle_layers.ctypes_data,
                self.cell_reverse_lookup.ctypes_data,
                self.cell_contents_count.ctypes_data,
                self.matrix.ctypes_data,
                self._cell_array.ctypes_data,
                self._boundary.ctypes_data,
                self._cell_edge_lengths.ctypes_data,
                self._positions.ctypes_data
                ]

        print self._p1_lib['LayerSort'](*args)














