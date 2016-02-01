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
        self._n_layers = 0

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

        self._n_layers = self.matrix.ncol
        self._n_cells = self.matrix.nrow
        

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
                     const int blocksize2[3],
                     const int threadsize2[3],
                     const int n,
                     const int nc,
                     int* nl,
                     int* n_cells,
                     int* __restrict__ d_pl,
                     int* __restrict__ d_crl,
                     int* __restrict__ d_ccc,
                     int** __restrict__ d_M,
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
        //__constant__ int d_nl;
        __constant__ int d_nc;


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
            const int C1 = (int)(( d_p[_ix*3 +1]  - d_b[2] ) / d_cel[1]);
            const int C2 = (int)(( d_p[_ix*3 +2]  - d_b[4] ) / d_cel[2]);

            const int val = (C2*d_ca[1] + C1)*d_ca[0] + C0;


            d_crl[_ix] = val;
            //old=atomicAdd(address, new);
            d_pl[_ix] = atomicAdd(&d_ccc[val], (int)1);


        }
        return;
        }

        __global__ void d_MaxLayers(const int* __restrict__ d_ccc, int * nl_out){
        
        const int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        int val = 0;

        if (_ix < d_nc){
            val = d_ccc[_ix];
        }
        
        for (int offset = warpSize/2; offset > 0; offset /=2){
            //val = fmaxf(val, __shfl_down(val,offset));
            int tmp = __shfl_down(val,offset);
            //val = (val > tmp) ? val : tmp;
            //asm("max.s32 %%0, %%1, %%2;" : "=r"(val) : "r"(val), "r"(tmp));
            val = max(val,tmp);
        }
        
        if ((int)(threadIdx.x & (warpSize - 1)) == 0){
            atomicMax(nl_out, val);
        }
        
        return;
        }

        __global__ void d_PopulateMatrix(const int d_nl,
                                         const int* __restrict__ d_pl,
                                         const int* __restrict__ d_crl,
                                         int* __restrict__ d_M
        ){

        const int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (_ix < d_n){
            
            d_M[ d_crl[_ix]*d_nl + d_pl[_ix]  ] = _ix;
        }
        return;
        }


        int LayerSort(%(ARGS)s){
            int err = 0;
            checkCudaErrors(cudaMemcpyToSymbol(d_n, &n, sizeof(n)));
            checkCudaErrors(cudaMemcpyToSymbol(d_nc, &nc, sizeof(nc)));
            //checkCudaErrors(cudaMemcpyToSymbol(d_nl, nl, sizeof(*nl)));

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];


            d_LayerSort<<<bs,ts>>>(d_pl, d_crl, d_ccc, d_ca, d_b, d_cel, d_p);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" d_LayerSort Execution failed. \\n");
            
            // bit of global memory for maximum number of layers.
            int * d_nl; cudaMalloc((void**)&d_nl, sizeof(int));
            cudaMemcpy(d_nl, nl, sizeof(int), cudaMemcpyHostToDevice);

            dim3 bs2; bs2.x = blocksize2[0]; bs2.y = blocksize2[1]; bs2.z = blocksize2[2];
            dim3 ts2; ts2.x = threadsize2[0]; ts2.y = threadsize2[1]; ts2.z = threadsize2[2];
            

            d_MaxLayers<<<bs2,ts2>>>(d_ccc, d_nl);
            int old_nl = *nl;
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" d_MaxLayers Execution failed. \\n");
            cudaMemcpy(nl, d_nl, sizeof(int), cudaMemcpyDeviceToHost);

            if ((*nl)*(*n_cells)>old_nl*(*n_cells)){
            //need to resize.
                cudaFree(*d_M);
                printf("new number of layers = %%d \\n", *nl);
                cudaMalloc((void**)d_M, (*nl)*(*n_cells)*sizeof(int));
            }

            //checkCudaErrors(cudaMemcpyToSymbol(d_nl, nl, sizeof(*nl)));

            d_PopulateMatrix<<<bs,ts>>>(*nl, d_pl, d_crl, *d_M);
            checkCudaErrors(cudaDeviceSynchronize());

            return err;
        }
        ''' % {'ARGS':p1_args}

        self._p1_lib = cuda_build.simple_lib_creator(_p1_header_code, _p1_code, 'CellOccupancyMatrix')

        self._init = True

    def sort(self):

        # Things that need to vary in size.
        if self.particle_layers.ncomp < self._n_func():
            self.particle_layers.realloc(self._n_func())
            self.cell_reverse_lookup.realloc(ncomp=self._n_func())

        if self.cell_contents_count.ncomp < self._domain.cell_count:
            self.cell_contents_count.realloc(self._domain.cell_count)
            self.matrix = cuda_base.Matrix(nrow=self._domain.cell_count, ncol=self.matrix.ncol)

        # Things that need to hold correct values.
        self._boundary.sync_from_version(self._domain.boundary_outer)
        self._cell_edge_lengths.sync_from_version(self._domain.cell_edge_lengths)
        self._cell_array.sync_from_version(self._domain.cell_array)

        self.cell_contents_count.zero()

        _tpb = 512
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(self._n_func() / float(_tpb))), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)

        _tpb2 = 128
        
        _ca = self._domain.cell_array
        _ca = _ca[0] * _ca[1] * _ca[2]

        _blocksize2 = (ctypes.c_int * 3)(int(math.ceil(_ca / float(_tpb))), 1, 1)
        _threadsize2 = (ctypes.c_int * 3)(_tpb, 1, 1)
        
        _nl = ctypes.c_int(self._n_layers)
        _n_cells = ctypes.c_int(self._n_cells)

        args = [_blocksize,
                _threadsize,
                _blocksize2,
                _threadsize2,
                ctypes.c_int(self._n_func()),
                ctypes.c_int(_ca),
                ctypes.byref(_nl),
                ctypes.byref(_n_cells),
                self.particle_layers.ctypes_data,
                self.cell_reverse_lookup.ctypes_data,
                self.cell_contents_count.ctypes_data,
                ctypes.byref(self.matrix.ctypes_data),
                self._cell_array.ctypes_data,
                self._boundary.ctypes_data,
                self._cell_edge_lengths.ctypes_data,
                self._positions.ctypes_data
                ]

        rval = self._p1_lib['LayerSort'](*args)
        
        self._n_layers = _nl.value
        self.matrix.ncol = self._n_layers

    @property
    def layers_per_cell(self):
        return self._n_layers

    @property
    def domain(self):
        return self._domain




# Default
OCCUPANCY_MATRIX = None









