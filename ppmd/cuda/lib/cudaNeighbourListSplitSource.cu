
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

__global__ void CreateNeighbourList(
    int* __restrict__ d_W,
    const double* __restrict__ d_positions,
    const int* __restrict__ d_CRL,
    const int* __restrict__ d_OM,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_CIHF
){

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

            if ( d_CIHF[cpp] < 1 ) {

                // loop over layers in neighbouring cell.
                for( int _idy = 0; _idy < d_ccc[cpp]; _idy++){

                    // get particle in cell cpp in layer _idy
                    const int idy = d_OM[cpp*d_nlayers + _idy];

                    //printf("NL ix=%d, iy=%d\n", idx, idy);

                    if (idx != idy){

                        const double3 r1 = {
                                            d_positions[idy*3    ] - r0.x,
                                            d_positions[idy*3 + 1] - r0.y,
                                            d_positions[idy*3 + 2] - r0.z
                                            };
                        if ( __fma_rn(r1.x, r1.x, __fma_rn(r1.y, r1.y, r1.z*r1.z)) < d_cutoff_squared ){
                            m++;
                            if (m < d_nmax){
                                d_W[idx + d_npart * m] = idy;
                            }
                        }
                    } //ix==iy check
                }//layers
            }

        } //directions

        //Number of neighbours for this particle

        d_W[idx] = (m<d_nmax) ? m : -1*m;


    } //threads

    return;
}


int NeighbourList(
    const int blocksize[3],
    const int threadsize[3],
    const int h_nmax,
    const int h_npart,
    const int h_nlayers,
    const double h_cutoff_squared,
    const int* __restrict__ h_CA,
    const cuda_Array<int> d_CIHF,
    const cuda_ParticleDat<double> d_positions,
    const cuda_Array<int> d_CRL,
    const cuda_Matrix<int> d_OM,
    const cuda_Array<int> d_ccc,
    cuda_Matrix<int> d_W
){

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

    CreateNeighbourList<<<bs,ts>>>(d_W.ptr, d_positions.ptr, d_CRL.ptr, d_OM.ptr, d_ccc.ptr, d_CIHF.ptr);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("NeighbourListLayerBased (split) library failed \n");

    int minfound = cudaMinElementInt(d_W.ptr, h_npart);
    return minfound;
}

__global__ void CreateNeighbourList2(
    int* __restrict__ d_W,
    const double* __restrict__ d_positions,
    const int* __restrict__ d_CRL,
    const int* __restrict__ d_OM,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_CIHF
){

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

        if (d_CIHF[cp] > -1)
        {
            //loop over 27 directions.
            for(int k = 0; k < 27; k++){

                // other cell.
                const int cpp = cp + d_cell_offsets[k];

                if ( d_CIHF[cpp] > 0 ) {

                    // loop over layers in neighbouring cell.
                    for( int _idy = 0; _idy < d_ccc[cpp]; _idy++){

                        // get particle in cell cpp in layer _idy
                        const int idy = d_OM[cpp*d_nlayers + _idy];

                        //printf("NL ix=%d, iy=%d\n", idx, idy);

                        if (idx != idy){

                            const double3 r1 = {
                                                d_positions[idy*3    ] - r0.x,
                                                d_positions[idy*3 + 1] - r0.y,
                                                d_positions[idy*3 + 2] - r0.z
                                                };
                            if ( __fma_rn(r1.x, r1.x, __fma_rn(r1.y, r1.y, r1.z*r1.z)) < d_cutoff_squared ){
                                m++;
                                if (m < d_nmax){
                                    d_W[idx + d_npart * m] = idy;
                                }
                            }
                        } //ix==iy check
                    }//layers
                }

            } //directions
        }
        //Number of neighbours for this particle

        d_W[idx] = (m<d_nmax) ? m : -1*m;


    } //threads

    return;
}


int NeighbourList2(
    const int blocksize[3],
    const int threadsize[3],
    const int h_nmax,
    const int h_npart,
    const int h_nlayers,
    const double h_cutoff_squared,
    const int* __restrict__ h_CA,
    const cuda_Array<int> d_CIHF,
    const cuda_ParticleDat<double> d_positions,
    const cuda_Array<int> d_CRL,
    const cuda_Matrix<int> d_OM,
    const cuda_Array<int> d_ccc,
    cuda_Matrix<int> d_W
){

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

    CreateNeighbourList2<<<bs,ts>>>(d_W.ptr, d_positions.ptr, d_CRL.ptr, d_OM.ptr, d_ccc.ptr, d_CIHF.ptr);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("NeighbourListLayerBased (split) library failed \n");

    int minfound;
    cudaError_t err = cudaMinElementInt(d_W.ptr, h_npart, &minfound);
    if (err != cudaSuccess) {getLastCudaError("NeighbourListLayerBased library failed \n");}
    getLastCudaError("NeighbourListLayerBased library failed \n");

    return minfound;
}



