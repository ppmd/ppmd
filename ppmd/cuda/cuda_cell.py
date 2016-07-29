"""
CUDA implementations of methods to handle the cell decomposition of a domain.
"""

#system
import ctypes
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


        self.update_required = True

        self._update_set = False
        self._update_func = None
        self._update_func_pre = None
        self._update_func_post = None


    def trigger_update(self):
        self.update_required = True


    def setup_update_tracking(self, func):
        """
        Setup an automatic cell update.
        :param func:
        """
        self._update_func = func
        self._update_set = True

    def setup_callback_on_update(self, func):
        """
        Setup a function to be ran after the cell list if updated.
        :param func: Function to run.
        :return:
        """
        self._update_func_post = func

    def setup_pre_update(self, func):
        self._update_func_pre = func

    def _update_tracking(self):

        if self._update_func is None:
            return True

        if self._update_set and self._update_func():
            return True
        else:
            return False

    def _pre_update(self):
        """
        Run a pre update function eg boundary conditions.
        """
        if self._update_func_pre is not None:
            self._update_func_pre()
            # pass

    def create(self):
        self._cell_sort_setup()


    def check(self):
        """
        Check if the cell_list needs updating and update if required.
        :return:
        """

        if not self._init:
            self._cell_sort_setup()

            if not self._init:
                print "Initalisation failed"
                return False


        if (self.update_required is True) or self._update_tracking():

            self._pre_update()

            self.sort()
            if self._update_func_post is not None:
                self._update_func_post()
            return True
        else:
            return False


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

    def _cell_sort_setup(self):

        self.particle_layers = cuda_base.Array(ncomp=self._n_func(), dtype=ctypes.c_int)
        self.cell_reverse_lookup = cuda_base.Array(ncomp=self._n_func(), dtype=ctypes.c_int)
        self.cell_contents_count = cuda_base.Array(ncomp=self._domain.cell_count, dtype=ctypes.c_int)
        self.matrix = cuda_base.device_buffer_2d(nrow=self._domain.cell_count,
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

            const int C0 = 1 + __double2int_rz(( d_p[_ix*3]    - d_b[0] ) / d_cel[0]);
            const int C1 = 1 + __double2int_rz(( d_p[_ix*3 +1]  - d_b[2] ) / d_cel[1]);
            const int C2 = 1 + __double2int_rz(( d_p[_ix*3 +2]  - d_b[4] ) / d_cel[2]);

            const int val = (C2*d_ca[1] + C1)*d_ca[0] + C0;

            //printf("COM ix=%%d, val=%%d, %%f, %%f, %%f\\n %%f, %%f, %%f\\n",
            //_ix, val, d_p[_ix*3], d_p[_ix*3+1], d_p[_ix*3+2], d_b[0], d_b[2], d_b[4] );







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

        #self._boundary.sync_from_version(self._domain.boundary)

        self._boundary[:] = self._domain.boundary[:]

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

    @property
    def positions(self):
        return self._positions



# Default
OCCUPANCY_MATRIX = None

class NeighbourListLayerBased(object):

    def __init__(self, occ_matrix=OCCUPANCY_MATRIX, cutoff=None):

        self._occ_matrix = occ_matrix
        assert cutoff is not None, "cuda_cell::NeighbourListLayerBased.setup error: No cutoff passed."
        self._rc = cutoff

        self.max_neigbours_per_particle = None


        self.list = cuda_base.Matrix(nrow=1, ncol=1, dtype=ctypes.c_int)

        _name = 'NeighbourList'

        hargs = '''
            const int blocksize[3],
            const int threadsize[3],
            const int h_nmax,
            const int h_npart,
            const int h_nlayers,
            const double h_cutoff_squared,
            const int* __restrict__ h_CA,
            const cuda_ParticleDat<double> d_positions,
            const cuda_Array<int> d_CRL,
            const cuda_Matrix<int> d_OM,
            const cuda_Array<int> d_ccc,
            cuda_Matrix<int> d_W
        '''

        dargs = '''
            int* __restrict__ d_W,
            const double* __restrict__ d_positions,
            const int* __restrict__ d_CRL,
            const int* __restrict__ d_OM,
            const int* __restrict__ d_ccc
        '''

        d_call_args = '''d_W.ptr, d_positions.ptr, d_CRL.ptr, d_OM.ptr, d_ccc.ptr'''


        _header = '''
            #include <cuda_generic.h>
            extern "C" int %(NAME)s(%(HARGS)s);
        ''' % {'NAME': _name, 'HARGS': hargs}


        _code = '''

            const int h_map[27][3] = {
                                {-1,1,-1},
                                {-1,-1,-1},
                                {-1,0,-1},
                                {0,1,-1},
                                {0,-1,-1},
                                {0,0,-1},
                                {1,0,-1},
                                {1,1,-1},
                                {1,-1,-1},

                                {-1,1,0},
                                {-1,0,0},
                                {-1,-1,0},
                                {0,-1,0},
                                {0,0,0},
                                {0,1,0},
                                {1,0,0},
                                {1,1,0},
                                {1,-1,0},

                                {-1,0,1},
                                {-1,1,1},
                                {-1,-1,1},
                                {0,0,1},
                                {0,1,1},
                                {0,-1,1},
                                {1,0,1},
                                {1,1,1},
                                {1,-1,1}
                              };


            __constant__ int d_nmax;                    //maximum number of neighbours
            __constant__ int d_npart;
            __constant__ int d_cell_offsets[27];
            __constant__ int d_nlayers;
            __constant__ double d_cutoff_squared;

            __global__ void CreateNeighbourList(%(DARGS)s){

                const int idx = threadIdx.x + blockIdx.x*blockDim.x;

                if (idx < d_npart){

                    // position of this particle
                    const double3 r0 = {d_positions[idx*3],
                                        d_positions[idx*3 + 1],
                                        d_positions[idx*3 + 2]};

                    //index of next neighbour
                    int m = 0;

                    //cell containing this particle
                    const int cp = d_CRL[idx];

                    //loop over 27 directions.
                    for(int k = 0; k < 27; k++){

                        // other cell.
                        const int cpp = cp + d_cell_offsets[k];


                        // loop over layers in neighbouring cell.
                        for( int _idy = 0; _idy < d_ccc[cpp]; _idy++){

                            // get particle in cell cpp in layer _idy
                            const int idy = d_OM[cpp*d_nlayers + _idy];

                            //printf("NL ix=%%d, iy=%%d\\n", idx, idy);

                            if (idx != idy){

                                const double3 r1 = {
                                                    d_positions[idy*3    ] - r0.x,
                                                    d_positions[idy*3 + 1] - r0.y,
                                                    d_positions[idy*3 + 2] - r0.z
                                                    };

                                // see if candidate neighbour

                                //printf("NL ix=%%d, iy=%%d, r2_fma=%%f, r2_trad=%%f \\n", idx, idy, __fma_rz(r1.x, r1.x, __fma_rz(r1.y, r1.y, r1.z*r1.z)),
                                // r1.x*r1.x + r1.y*r1.y + r1.z*r1.z);

                                if ( __fma_rn(r1.x, r1.x, __fma_rn(r1.y, r1.y, r1.z*r1.z)) < d_cutoff_squared ){

                                    // work out new index
                                    m++;


                                    //printf("NL ix=%%d : d_W[%%d] = %%d \\n", idx, idx + d_npart * m, idy);
                                    d_W[idx + d_npart * m] = idy;

                                    //experiment to swap order, appears to be faster by
                                    // about 10%%
                                    //d_W[idx*d_nmax + m] = idy;
                                    //d_W[idx] = idy + m;

                                }


                            } //ix==iy check
                        }//layers
                    } //directions

                    //Number of neighbours for this particle
                    d_W[idx] = m;

                } //threads

                return;
            }


            int %(NAME)s(%(HARGS)s){

                int h_offsets[27];

                for(int ix = 0; ix < 27; ix++){
                    h_offsets[ix] = h_map[ix][0] + h_map[ix][1] * h_CA[0] + h_map[ix][2] * h_CA[0]*h_CA[1];
                }


                checkCudaErrors(cudaMemcpyToSymbol(d_cell_offsets, &h_offsets[0], 27*sizeof(int)));
                checkCudaErrors(cudaMemcpyToSymbol(d_nmax, &h_nmax, sizeof(int)));
                checkCudaErrors(cudaMemcpyToSymbol(d_npart, &h_npart, sizeof(int)));
                checkCudaErrors(cudaMemcpyToSymbol(d_nlayers, &h_nlayers, sizeof(int)));
                checkCudaErrors(cudaMemcpyToSymbol(d_cutoff_squared, &h_cutoff_squared, sizeof(double)));


                dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
                dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];

                CreateNeighbourList<<<bs,ts>>>(%(D_CALL_ARGS)s);
                checkCudaErrors(cudaDeviceSynchronize());
                getLastCudaError("NeighbourListLayerBased library failed \\n");

                return 0;
            }
        ''' % {'NAME': _name, 'HARGS': hargs, 'DARGS': dargs, 'D_CALL_ARGS': d_call_args}

        self._lib = cuda_build.simple_lib_creator(_header, _code, _name)[_name]



    def update(self):

        if (self.list.ncol < self._occ_matrix.positions.npart_local) or (self.list.nrow < (8 * self._occ_matrix.layers_per_cell + 1)):
            self.list.realloc(nrow=8 * self._occ_matrix.layers_per_cell + 1,
                              ncol=self._occ_matrix.positions.npart_local)

        _tpb = 256
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(self._occ_matrix.positions.npart_local / float(_tpb))), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)


        args = (
            _blocksize,
            _threadsize,
            ctypes.c_int(8 * self._occ_matrix.layers_per_cell + 1), # nmax
            ctypes.c_int(self._occ_matrix.positions.npart_local),      # npart
            ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
            ctypes.c_double(self._rc ** 2),                      # cutoff squared
            self._occ_matrix.domain.cell_array.ctypes_data,
            self._occ_matrix.positions.struct,
            self._occ_matrix.cell_reverse_lookup.struct,
            self._occ_matrix.matrix.struct,
            self._occ_matrix.cell_contents_count.struct,
            self.list.struct
        )

        self._lib(*args)
        self.max_neigbours_per_particle = 8 * self._occ_matrix.layers_per_cell + 1































