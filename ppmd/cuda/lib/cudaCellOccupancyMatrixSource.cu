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
            //asm("max.s32 %0, %1, %2;" : "=r"(val) : "r"(val), "r"(tmp));
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


        int LayerSort(const int f_MPI_COMM,
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
                     ){
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

            if (n > 0){
                d_LayerSort<<<bs,ts>>>(d_pl, d_crl, d_ccc, d_p);
            }

            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" d_LayerSort Execution failed. \n");
            
            // bit of global memory for maximum number of layers.
            int * d_nl; cudaMalloc((void**)&d_nl, sizeof(int));
            cudaMemcpy(d_nl, nl, sizeof(int), cudaMemcpyHostToDevice);

            dim3 bs2; bs2.x = blocksize2[0]; bs2.y = blocksize2[1]; bs2.z = blocksize2[2];
            dim3 ts2; ts2.x = threadsize2[0]; ts2.y = threadsize2[1]; ts2.z = threadsize2[2];
            

            d_MaxLayers<<<bs2,ts2>>>(d_ccc, d_nl);
            int old_nl = *nl;
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" d_MaxLayers Execution failed. \n");

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
                //printf("new number of layers = %d, number of cells %d \n", *nl, *n_cells);
                checkCudaErrors(cudaMalloc((void**)d_M, (*nl)*(*n_cells)*sizeof(int)));

                /*
                printf("new pointer %ld \n", (long)(*d_M));
                int tmp;
                int err=cudaMemcpy(&tmp, *d_M + 511, sizeof(int), cudaMemcpyDeviceToHost);
                printf("err %d, tmp %d \n", err, tmp);
                */

            }

            //checkCudaErrors(cudaMemcpyToSymbol(d_nl, nl, sizeof(*nl)));
            if (n > 0){
                d_PopulateMatrix<<<bs,ts>>>(*nl, d_pl, d_crl, *d_M);
            }
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
                    const int col = ix % d_old_ncol;
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




        