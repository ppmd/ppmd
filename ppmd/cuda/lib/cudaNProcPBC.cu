


#include "cudaNProcPBCSource.h"


__constant__ int d_bin_to_dir[57];
__constant__ double d_shifts[78];
__constant__ double B[6];


__global__ void d_find_escapees(
    const int d_n,
    double * __restrict__ P,
    int * __restrict__ d_count,
    int * __restrict__ d_dir_count,
    int * __restrict__ d_escapees
){
    const int _ix = threadIdx.x + blockIdx.x*blockDim.x;
    if (_ix < d_n) {

        int b = 0;

        //Check x direction
        if (P[3*_ix] < B[0]){
            b ^= 4;
        }else if (P[3*_ix] >= B[1]){
            b ^= 32;
        }

        //check y direction
        if (P[3*_ix+1] < B[2]){
            b ^= 2;
        }else if (P[3*_ix+1] >= B[3]){
            b ^= 16;
        }

        //check z direction
        if (P[3*_ix+2] < B[4]){
            b ^= 1;
        }else if (P[3*_ix+2] >= B[5]){
            b ^= 8;
        }

        //If b > 0 then particle has escaped through some boundary
        if (b>0){

            const int dir = d_bin_to_dir[b];

            d_P[3*_ix] += d_shifts[3*dir];
            d_P[3*_ix+1] += d_shifts[3*dir+1];
            d_P[3*_ix+2] += d_shifts[3*dir+2];

            const int jx = 3 * atomicAdd(d_count, (int)1);
            const int dx = atomicAdd(&d_dir_count[dir], (int)1);

            d_escapees[jx] = dir;
            d_escapees[jx+1] = _ix;
            d_escapees[jx+2] = dx;

        }
    }
    return;
}




int cudaNProcPBCStageOne(
    const int h_n,
    const double * __restrict__ h_B,
    double * __restrict__ d_P,
    const * __restrict__ h_shifts,
    int * __restrict__ d_count,
    int * __restrict__ d_dir_count,
    int * __restrict__ d_escapees
){
    cudaError_t err;
    dim3 bs, ts;


    int bin_to_dir[57] = {0};
    bin_to_dir[ 7  ]=0;
    bin_to_dir[ 3  ]=1;
    bin_to_dir[ 35 ]=2;
    bin_to_dir[ 5  ]=3;
    bin_to_dir[ 1  ]=4;
    bin_to_dir[ 33 ]=5;
    bin_to_dir[ 21 ]=6;
    bin_to_dir[ 17 ]=7;
    bin_to_dir[ 49 ]=8;

    bin_to_dir[ 6  ]=9;
    bin_to_dir[ 2  ]=10;
    bin_to_dir[ 34 ]=11;
    bin_to_dir[ 4  ]=12;
    bin_to_dir[ 32 ]=13;
    bin_to_dir[ 20 ]=14;
    bin_to_dir[ 16 ]=15;
    bin_to_dir[ 48 ]=16;

    bin_to_dir[ 14 ]=17;
    bin_to_dir[ 10 ]=18;
    bin_to_dir[ 42 ]=19;
    bin_to_dir[ 12 ]=20;
    bin_to_dir[ 8  ]=21;
    bin_to_dir[ 40 ]=22;
    bin_to_dir[ 28 ]=23;
    bin_to_dir[ 24 ]=24;
    bin_to_dir[ 56 ]=25;

    err = cudaMemcpyToSymbol(d_bin_to_dir, &bin_to_dir[0], 57*sizeof(int));
    if(err>0){return err;}
    err = cudaMemcpyToSymbol(d_shifts, &h_shifts[0], 78*sizeof(double));
    if(err>0){return err;}
    err = cudaMemcpyToSymbol(B, h_B, 6*sizeof(double));
    if(err>0){return err;}
    err = cudaCreateLaunchArgs(h_n, 256, &bs, &ts);
    if(err>0){return err;}

    d_find_escapees<<<bs, ts>>>(h_n, d_P, d_count, d_dir_count, d_escapees);
    err = cudaDeviceSynchronize();
    if(err>0){return err;}

    err = cudaGetLastError();
    return err;
}


__global__ void d_PopulateEscapeMatrix(
    const int d_n,
    const int d_ncol,
    const int * __restrict__ d_escape_list,
    int * __restrict__ d_escape_matrix
){
    const int ix = threadIdx.x + blockIdx.x*blockDim.x;

    if (ix<d_n) {
        const int row = d_escape_list[3*ix];
        const int id = d_escape_list[3*ix+1];
        const int col = d_escape_list[3*ix+2];
        d_escape_matrix[row*d_ncol + col] = id;
    }

    return;
}



int cudaNProcPBCStageTwo(
    const int h_n,
    const int d_ncol,
    const int * __restrict__ d_escape_list,
    int * __restrict__ d_escape_matrix
){
    cudaError_t err;
    dim3 bs, ts;

    err = cudaCreateLaunchArgs(h_n, 256, &bs, &ts);
    if(err>0){return err;}

    d_PopulateEscapeMatrix<<<bs, ts>>>(h_n,
                                       d_ncol,
                                       d_escape_list,
                                       d_escape_matrix);
    err = cudaDeviceSynchronize();
    if(err>0){return err;}

    err = cudaGetLastError();
    return err;
}







