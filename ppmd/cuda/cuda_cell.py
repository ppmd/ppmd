"""
CUDA implementations of methods to handle the cell decomposition of a domain.
"""

#system
import ctypes
import math

#package
import ppmd.mpi as mpi

#cuda
import cuda_runtime
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
        self._p1_lib = None
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

        self.version_id = 0


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

    def reset_callbacks(self):
        self._update_func = None
        self._update_func_pre = None
        self._update_func_post = None


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
        

        #self._boundary = cuda_base.Array(initial_value=self._domain.boundary_outer)
        #self._cell_edge_lengths = cuda_base.Array(initial_value=self._domain.cell_edge_lengths)
        #self._cell_array = cuda_base.Array(initial_value=self._domain.cell_array, dtype=ctypes.c_int)



        self._setup = True
        self._build()


    def _build(self):
        """
        Build the library to create the cell occupancy matrix.
        :return:
        """
        assert self._setup is not False, "Run CellOccupancyMatrix.setup() first."

        p1_args = '''const int f_MPI_COMM,
                     const int MPI_FLAG,
                     const int blocksize[3],
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
                     const int* __restrict__ h_ca,
                     const double* __restrict__ h_b,
                     const double* __restrict__ h_cel,
                     const double* __restrict__ d_p
                     '''

        _p1_header_code = '''
        //Header
        #include <cuda_generic.h>
        #include <mpi.h>

        extern "C" int LayerSort(%(ARGS)s);
        extern "C" int copy_matrix_cols(const int h_old_ncol,
                                        const int h_new_ncol,
                                        const int h_nrow,
                                        const int * __restrict__ d_old_ptr,
                                        int * __restrict__ d_new_ptr);

        ''' %{'ARGS': p1_args}

        _p1_code = '''
        //source

        __constant__ int d_n;
        //__constant__ int d_nl;
        __constant__ int d_nc;

        __constant__ double _icel0;
        __constant__ double _icel1;
        __constant__ double _icel2;

        __constant__ double _b0;
        __constant__ double _b2;
        __constant__ double _b4;
        __constant__ double _b1;
        __constant__ double _b3;
        __constant__ double _b5;

        __constant__ double _ca0;
        __constant__ double _ca1;
        __constant__ double _ca2;

        __global__ void d_LayerSort(int* __restrict__ d_pl,
                                    int* __restrict__ d_crl,
                                    int* __restrict__ d_ccc,
                                    const double* __restrict__ d_p
        ){

        const int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (_ix < d_n){

            int C0 = 1 + __double2int_rz(( d_p[_ix*3]    - _b0 )*_icel0);
            int C1 = 1 + __double2int_rz(( d_p[_ix*3+1]  - _b2 )*_icel1);
            int C2 = 1 + __double2int_rz(( d_p[_ix*3+2]  - _b4 )*_icel2);

            if ( (C0 > (_ca0-2)) && (d_p[_ix*3]   <= _b1 )) {C0 = _ca0-2;}
            if ( (C1 > (_ca1-2)) && (d_p[_ix*3+1] <= _b3 )) {C1 = _ca1-2;}
            if ( (C2 > (_ca2-2)) && (d_p[_ix*3+2] <= _b5 )) {C2 = _ca2-2;}

            const int val = (C2*_ca1 + C1)*_ca0 + C0;

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

            MPI_Comm MPI_COMM;
            if (MPI_FLAG > 0){
                MPI_COMM = MPI_Comm_f2c(f_MPI_COMM);
            }

            const double _hicel0 = 1.0/h_cel[0];
            const double _hicel1 = 1.0/h_cel[1];
            const double _hicel2 = 1.0/h_cel[2];

            const double _hb0 = h_b[0];
            const double _hb2 = h_b[2];
            const double _hb4 = h_b[4];
            const double _hb1 = h_b[1];
            const double _hb3 = h_b[3];
            const double _hb5 = h_b[5];

            const double _hca0 = h_ca[0];
            const double _hca1 = h_ca[1];
            const double _hca2 = h_ca[2];

            checkCudaErrors(cudaMemcpyToSymbol(_icel0, &_hicel0, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_icel1, &_hicel1, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_icel2, &_hicel2, sizeof(double)));

            checkCudaErrors(cudaMemcpyToSymbol(_b0, &_hb0, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_b2, &_hb2, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_b4, &_hb4, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_b1, &_hb1, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_b3, &_hb3, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_b5, &_hb5, sizeof(double)));

            checkCudaErrors(cudaMemcpyToSymbol(_ca0, &_hca0, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_ca1, &_hca1, sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(_ca2, &_hca2, sizeof(double)));


            checkCudaErrors(cudaMemcpyToSymbol(d_n, &n, sizeof(n)));
            checkCudaErrors(cudaMemcpyToSymbol(d_nc, &nc, sizeof(nc)));
            //checkCudaErrors(cudaMemcpyToSymbol(d_nl, nl, sizeof(*nl)));

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];


            d_LayerSort<<<bs,ts>>>(d_pl, d_crl, d_ccc, d_p);
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

            int tnl = 0;
            cudaMemcpy(&tnl, d_nl, sizeof(int), cudaMemcpyDeviceToHost);

            if(MPI_FLAG > 0){
                MPI_Allreduce(&tnl, nl, 1, MPI_INT, MPI_MAX, MPI_COMM);
            } else {
                *nl = tnl;
            }

            if ((*nl)*(*n_cells)>old_nl*(*n_cells)){
            //need to resize.
                cudaFree(*d_M);
                //printf("new number of layers = %%d, number of cells %%d \\n", *nl, *n_cells);
                checkCudaErrors(cudaMalloc((void**)d_M, (*nl)*(*n_cells)*sizeof(int)));

                /*
                printf("new pointer %%ld \\n", (long)(*d_M));
                int tmp;
                int err=cudaMemcpy(&tmp, *d_M + 511, sizeof(int), cudaMemcpyDeviceToHost);
                printf("err %%d, tmp %%d \\n", err, tmp);
                */

            }

            //checkCudaErrors(cudaMemcpyToSymbol(d_nl, nl, sizeof(*nl)));

            d_PopulateMatrix<<<bs,ts>>>(*nl, d_pl, d_crl, *d_M);
            checkCudaErrors(cudaDeviceSynchronize());

            return err;
        }


        // ---------- realloc matrix code --------------

        __global__ void copy_matrix_cols_kernel(
            const int d_n,
            const int d_old_ncol,
            const int d_new_ncol,
            const int d_nrow,
            const int * __restrict__ d_old_ptr,
            int * __restrict__ d_new_ptr
        ){
            const int ix = threadIdx.x + blockIdx.x * blockDim.x;
            if (ix<d_n){
                    const int row = ix/d_old_ncol;
                    const int col = ix %% d_old_ncol;
                    const int val = d_old_ptr[row*d_old_ncol + col];
                    d_new_ptr[row*d_new_ncol + col] = val;
            }
            return;
        }

        int copy_matrix_cols(
            const int h_old_ncol,
            const int h_new_ncol,
            const int h_nrow,
            const int * __restrict__ d_old_ptr,
            int * __restrict__ d_new_ptr
        ){
            cudaError_t err;
            dim3 bs, ts;
            const int h_n = h_old_ncol*h_nrow;
            err = cudaCreateLaunchArgs(h_n, 1024, &bs, &ts);
            if(err>0){return err;}
            copy_matrix_cols_kernel<<<bs,ts>>>(h_n,
                                               h_old_ncol,
                                               h_new_ncol,
                                               h_nrow,
                                               d_old_ptr,
                                               d_new_ptr);
            err = cudaDeviceSynchronize();
            return err;

        }




        ''' % {'ARGS':p1_args}

        self._p1_lib = cuda_build.simple_lib_creator(
            _p1_header_code, _p1_code, 'CellOccupancyMatrix')

        self._init = True

    def sort(self):


        # Things that need to vary in size.
        if self.particle_layers.ncomp < self._n_func():
            self.particle_layers.realloc(self._n_func())
            self.cell_reverse_lookup.realloc(ncomp=self._n_func())

        if self.cell_contents_count.ncomp < self._domain.cell_count:
            self.cell_contents_count.realloc(self._domain.cell_count)
            self.matrix.realloc(nrow=self._domain.cell_count,
                                ncol=self.matrix.ncol)

        # Things that need to hold correct values.

        #self._boundary.sync_from_version(self._domain.boundary)

        #self._boundary[:] = self._domain.boundary[:]

        #self._cell_edge_lengths.sync_from_version(self._domain.cell_edge_lengths)
        #self._cell_array.sync_from_version(self._domain.cell_array)

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

        args = [ctypes.c_int32(mpi.MPI_HANDLE.fortran_comm),
                ctypes.c_int(1),
                _blocksize,
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
                self._domain.cell_array.ctypes_data,
                self._domain.boundary.ctypes_data,
                self._domain.cell_edge_lengths.ctypes_data,
                self._positions.ctypes_data
                ]

        rval = self._p1_lib['LayerSort'](*args)

        # print "layersort out pointer", self.matrix.ctypes_data


        self._n_layers = _nl.value
        self.matrix.ncol = self._n_layers

        self.version_id += 1
        self.update_required = False

    @property
    def layers_per_cell(self):
        return self._n_layers

    @property
    def domain(self):
        return self._domain

    @property
    def positions(self):
        return self._positions

    def prepare_halo_sort(self, max_halo_layers=None):
        assert max_halo_layers is not None, "no size passed"

        # Is a resize needed?
        if max_halo_layers > self._n_layers:


            new_matrix = cuda_base.device_buffer_2d(nrow=self.matrix.nrow,
                                                    ncol=max_halo_layers,
                                                    dtype=ctypes.c_int32)

            cuda_runtime.cuda_err_check(
            self._p1_lib['copy_matrix_cols'](
                                             ctypes.c_int32(self.matrix.ncol),
                                             ctypes.c_int32(new_matrix.ncol),
                                             ctypes.c_int32(new_matrix.nrow),
                                             self.matrix.ctypes_data,
                                             new_matrix.ctypes_data
                                            ))

            self.matrix.free()
            self.matrix = new_matrix


            self._n_layers = max_halo_layers







# Default
OCCUPANCY_MATRIX = None

class NeighbourListLayerBased(object):

    def __init__(self, occ_matrix=OCCUPANCY_MATRIX, cutoff=None):

        self._occ_matrix = occ_matrix
        assert cutoff is not None, "cuda_cell::NeighbourListLayerBased.setup error: No cutoff passed."
        self._rc = cutoff

        self.max_neigbours_per_particle = None
        self.version_id = 0

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

        self.version_id = self._occ_matrix.version_id































