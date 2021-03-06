#include "cudaMPILib.h"








__global__ void cudaLInfNorm_k_int(
    const int * __restrict__ d_ptr,
    const int len,
    int * __restrict__ d_val
){

    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    int t = 0;
    if (ix<len){
        t = (int) abs(d_ptr[ix]);
    }

    t = warpReduceMax(t);

    __shared__ int dt[1];

    if (  (int)(threadIdx.x & (warpSize - 1)) == 0){
      dt[0] = 0;
    }

    __syncthreads();


    if (  (int)(threadIdx.x & (warpSize - 1)) == 0){
        atomicMax(&dt[0], t);
    }
    __syncthreads();

    if (threadIdx.x == 0){
         atomicMax(&d_val[0], dt[0]);
    }

    return;
}


cudaError_t cudaLInfNormInt(
    const int * __restrict__ d_ptr,
    const int len,
    int *val
)
{
    cudaError_t err;
    dim3 bs, ts;
    *val = 0;

    int *d_val;
    err = cudaMalloc(&d_val, sizeof(int));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpy(d_val, val, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {return err;}

    err = cudaCreateLaunchArgs(len, 512, &bs, &ts);
    if (err != cudaSuccess) { return err; }

    cudaLInfNorm_k_int<<<bs, ts>>>(
        d_ptr,
        len,
        d_val
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return err;}

    return cudaMemcpy(val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
}




// DIrectly based on MPI

int MPIErrorCheck_cuda(const int error_code){

    int err = 0;
    if (error_code != MPI_SUCCESS) {

       char error_string[BUFSIZ];
       int length_of_error_string;

       MPI_Error_string(error_code, error_string, &length_of_error_string);
       //fprintf(stderr, "%3d: %s\n", my_rank, error_string);
       cout << error_string << endl;

       err = 1;
    }

    return err;
}





int MPI_Bcast_cuda(const int FCOMM, void* buffer, const int byte_count, const int root){
    MPI_Comm COMM = MPI_Comm_f2c(FCOMM);
    MPI_Errhandler_set(COMM, MPI_ERRORS_RETURN);


    const int err = MPI_Bcast( buffer,
                               byte_count,
                               MPI_BYTE,
                               root,
                               COMM
                             );


    return err;
}


int MPI_Gatherv_cuda(const int FCOMM,
                     const void* s_buffer,
                     const int s_count,
                     void* r_buffer,
                     const int* r_counts,
                     const int* r_disps,
                     const int root
                     ){

    MPI_Comm COMM = MPI_Comm_f2c(FCOMM);

    MPI_Errhandler_set(COMM, MPI_ERRORS_RETURN);
    const int err = MPI_Gatherv(s_buffer,
                                s_count,
                                MPI_BYTE,
                                r_buffer,
                                r_counts,
                                r_disps,
                                MPI_BYTE,
                                root,
                                COMM);

    return err;
}














// MPI related

namespace _cudaFindNewSlots
{
    __constant__ int d_n1;
    __constant__ int d_n2;


    __global__ void cudaFindNewSlots_kernel1(const int * __restrict__ d_scan,
                                             int * __restrict__ d_empties ){

        int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (_ix < d_n1){
            if (d_scan[_ix] < d_scan[_ix+1]){
                 d_empties[d_scan[_ix]] = _ix;
            }
        }
        return;
    }


    __global__ void cudaFindNewSlots_kernel2(
        const int * __restrict__ d_scan,
        int * __restrict__ d_sources
    ){

        int _ix = threadIdx.x + blockIdx.x*blockDim.x;

        if (_ix < d_n2){

            const int ix = _ix + d_n1;
            const int num_empty_after_end = d_scan[ix] - d_scan[d_n1];

            //if non empty
            if (d_scan[ix] == d_scan[ix+1]){
                d_sources[_ix-num_empty_after_end] = ix;
            }

        }
        return;
    }



}


int cudaFindEmptySlots(const int blocksize1[3],
                       const int threadsize1[3],
                       const int* d_scan,
                       const int h_n1,
                       int * d_empties
                       ){

    //device constant copy.
    checkCudaErrors(cudaMemcpyToSymbol(_cudaFindNewSlots::d_n1, &h_n1, sizeof(int)));


    dim3 bs; bs.x = blocksize1[0]; bs.y = blocksize1[1]; bs.z = blocksize1[2];
    dim3 ts; ts.x = threadsize1[0]; ts.y = threadsize1[1]; ts.z = threadsize1[2];

    _cudaFindNewSlots::cudaFindNewSlots_kernel1<<<bs,ts>>>(d_scan, d_empties);

    return (int) cudaDeviceSynchronize();
}



int cudaFindNewSlots(const int blocksize2[3],
                     const int threadsize2[3],
                     const int* d_scan,
                     const int h_n1,
                     const int h_n2,
                     int * d_sources
                     ){

    //device constant copy.
    checkCudaErrors(cudaMemcpyToSymbol(_cudaFindNewSlots::d_n1, &h_n1, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(_cudaFindNewSlots::d_n2, &h_n2, sizeof(int)));

    dim3 bs2; bs2.x = blocksize2[0]; bs2.y = blocksize2[1]; bs2.z = blocksize2[2];
    dim3 ts2; ts2.x = threadsize2[0]; ts2.y = threadsize2[1]; ts2.z = threadsize2[2];

    _cudaFindNewSlots::cudaFindNewSlots_kernel2<<<bs2,ts2>>>(d_scan, d_sources);

    return (int) cudaDeviceSynchronize();

}








namespace _ExSizes
{   

    __global__ void GatherBoundaryCellCounts(
        const int* __restrict__ D_CCC,      // Cell contents count array
        const int n,                        // Number of cells to inspect
        const int* __restrict__ D_b_arr,    // actual indices of boundary cells
        int* D_b_tmp,                       // space to place boundary cell counts
        int* D_tmp_count                    // reduce the count accross cells into here
        ){


        int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        int cc = 0;

        if (_ix < n){
            const int C = D_CCC[D_b_arr[_ix]];
            D_b_tmp[_ix] = C;   // copy this cell count
            cc = C;             // set local tt to this cell count
        }

        // reduce tmp count accross warp
        cc = warpReduceSum(cc);

        // reduce into global mem
        if (threadIdx.x == 0){
            atomicAdd(D_tmp_count, cc);
        }

        return;
    }



    __global__ void ScatterHaloCellCounts(
        int* __restrict__ D_CCC,      // Cell contents count array
        const int n,                        // Number of cells to inspect
        const int* __restrict__ D_h_arr,    // actual indices of halo cells
        int* D_h_tmp,                       // space to place halo cell counts
        int* D_tmp_count                    // reduce the count across cells into here
        ){


        int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        int cc = 0;

        if (_ix < n){
            const int C = D_h_tmp[_ix];
            D_CCC[D_h_arr[_ix]] = C;        // add the entry to the local cell counts
            cc = C;                         // set local tt to this cell count
        }

        // reduce tmp count accross warp
        cc = warpReduceSum(cc);

        // reduce into global mem
        if (threadIdx.x == 0){
            atomicAdd(D_tmp_count, cc);
        }

        return;
    }


        __global__ void GatherScatterCellCounts(
        int* __restrict__ D_CCC,            // Cell contents count array
        const int n,                        // Number of cells to inspect
        const int* __restrict__ D_h_arr,    // actual indices of halo cells
        const int* __restrict__ D_b_arr,    // actual indices of boundary cells
        int* D_tmp_count                    // reduce the count across cells into here
        ){

        int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        int cc = 0;

        if (_ix < n){
            const int C = D_CCC[D_b_arr[_ix]];
            D_CCC[D_h_arr[_ix]] = C;        // add the entry to the local cell counts
            cc = C;                         // set local tt to this cell count
        }

        // reduce tmp count accross warp
        cc = warpReduceSum(cc);

        // reduce into global mem
        if (threadIdx.x == 0){
            atomicAdd(D_tmp_count, cc);
        }

        return;
    }



}

int cudaExchangeCellCounts(
        const int FCOMM,                        // Fortran communicator
        const int* __restrict__ H_SEND_RANKS,   // send ranks
        const int* __restrict__ H_RECV_RANKS,   // recv ranks 
        const int* __restrict__ H_h_ind,        // The starting indices for the halo cells
        const int* __restrict__ H_b_ind,        // The starting indices for the bound cellsi
        const int* __restrict__ D_h_arr,        // The halo cell indices
        const int* __restrict__ D_b_arr,        // The boundary cell indices
        int* __restrict__ D_CCC,                // Cell contents count array
        int* __restrict__ H_halo_count,         // RETURN: Number of halo particles
        int* __restrict__ H_tmp_count,          // RETURN: Amount of temporary space needed
        int* __restrict__ D_h_tmp,              // Temp storage for halo counts
        int* __restrict__ D_b_tmp,              // Temp storage for boundary counts
        int* __restrict__ H_dir_counts          // RETURN: Total expected recv counts per dir
        )
{   

    // var to use for errors
    int err;

    // vars for blocks and threads
    dim3 bs, ts;

    // MPI initialisations
    MPI_Comm COMM = MPI_Comm_f2c(FCOMM);
    int rank; MPI_Comm_rank(COMM, &rank);
    MPI_Status MPI_STATUS;

    
    //reset the return counts
    *H_tmp_count = 0;
    *H_halo_count = 0;

    const int const_0 = 0;

    int* D_tmp_count;
    int H_tmp_int;
    
    // make a device tmp
    err = (int) cudaMalloc(&D_tmp_count, sizeof(int)); 
    if (err != cudaSuccess) { return err; }
    
    // ensure is zero
    err = (int) cudaMemcpy(D_tmp_count, &const_0, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { return err; }



    for(int dir=0 ; dir<6 ; dir++ ){

        // Here we want to collect the local cell counts for a direction on the device
        // exchange these sizes and get the total for the direction

        const int dir_cell_count = H_b_ind[dir+1] - H_b_ind[dir];


    
        if (rank == H_RECV_RANKS[dir]){


            err = cudaCreateLaunchArgs(dir_cell_count, 512, &bs, &ts);
            if (err != cudaSuccess) { return err; }
            _ExSizes::GatherScatterCellCounts<<<bs, ts>>>(D_CCC,
                                                          dir_cell_count,
                                                          D_h_arr+H_h_ind[dir],
                                                          D_b_arr+H_b_ind[dir],
                                                          D_tmp_count);

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { return err; }

            err = (int) cudaMemcpy(&H_tmp_int, D_tmp_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { return err; }

            // Update amount of packing space that will be needed
            *H_tmp_count = MAX(*H_tmp_count, H_tmp_int);


        } else {

            const int dir_r_cell_count = H_h_ind[dir+1] - H_h_ind[dir];

            err = cudaCreateLaunchArgs(dir_cell_count, 512, &bs, &ts);
            if (err != cudaSuccess) { return err; }

            // zero cell count for this dir
            err = (int) cudaMemcpy(D_tmp_count, &const_0, sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { return err; }

            // Sum and pack cell counts

            err = cudaCreateLaunchArgs(dir_cell_count, 512, &bs, &ts);
            if (err != cudaSuccess) { return err; }
            _ExSizes::GatherBoundaryCellCounts<<<bs,ts>>>(D_CCC,
                                                          dir_cell_count,
                                                          D_b_arr+H_b_ind[dir],
                                                          D_b_tmp,
                                                          D_tmp_count);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { return err; }

            err = (int) cudaMemcpy(&H_tmp_int, D_tmp_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { return err; }

            // Update amount of packing space that will be needed
            *H_tmp_count = MAX(*H_tmp_count, H_tmp_int);





            MPI_Sendrecv (D_b_tmp, dir_cell_count, MPI_INT,
                          H_SEND_RANKS[dir], rank,
                          D_h_tmp, dir_r_cell_count, MPI_INT,
                          H_RECV_RANKS[dir], H_RECV_RANKS[dir],
                          COMM, &MPI_STATUS);





            // zero cell count for this dir
            err = (int) cudaMemcpy(D_tmp_count, &const_0, sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { return err; }

            // Sum and pack cell counts
            err = cudaCreateLaunchArgs(dir_r_cell_count, 512, &bs, &ts);
            if (err != cudaSuccess) { return err; }
            _ExSizes::GatherBoundaryCellCounts<<<bs,ts>>>(D_CCC,
                                                          dir_r_cell_count,
                                                          D_h_arr+H_h_ind[dir],
                                                          D_h_tmp,
                                                          D_tmp_count);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { return err; }

            err = (int) cudaMemcpy(&H_tmp_int, D_tmp_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { return err; }

            // Update amount of packing space that will be needed
            *H_tmp_count = MAX(*H_tmp_count, H_tmp_int);

        } // end of rank == H_RECV_RANK if

    }


    return 0;
}


namespace _cudaHaloArrayCopyScan
{

    __global__ void masked_copy(
        const int length,
        const int* __restrict__ d_map,
        const int* __restrict__ d_ccc,
        int* __restrict__ d_scan
    ){
        const int ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (ix < length){
             d_scan[ix] = d_ccc[d_map[ix]];
        }
        return;
    }

}













int cudaHaloArrayCopyScan(
    const int length,
    const int* __restrict__ d_map,
    const int* __restrict__ d_ccc,
    int* __restrict__ d_scan,
    int* __restrict__ h_max
){

    if (length == 0){ return 0; }

    dim3 bs, ts;
    cudaError_t err;

    err = cudaCreateLaunchArgs(length, 1024, &bs, &ts);
    if (err != cudaSuccess) { return err; }
    _cudaHaloArrayCopyScan::masked_copy<<<bs,ts>>>(length, d_map, d_ccc, d_scan);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { return err; }
    
    // thrust library is failing
    // err = cudaMaxElementInt(d_scan, length, h_max);
    err = cudaLInfNormInt(d_scan, length, h_max);
    if (err != cudaSuccess) { return err; }

    cudaExclusiveScanInt(d_scan, length+1);


    return 0;
}

namespace _cudaHaloFillOccupancyMatrix
{

    __global__ void fill_occ_matrix(
        const int length,
        const int max_count,
        const int occ_matrix_stride,
        const int n_local,
        const int* __restrict__ d_halo_indices,
        const int* __restrict__ d_ccc,
        const int* __restrict__ d_halo_scan,
        int* __restrict__ d_occ_matrix
    ){
        const int ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (ix < length){

            const int cid = ix/max_count;
            const int cx = d_halo_indices[cid];
            const int lx = ix % max_count;
            if (lx < d_ccc[cx]){

                const int offset = d_halo_scan[cid];
                d_occ_matrix[cx * occ_matrix_stride + lx] = n_local + offset + lx;

                /*
                if ( cx * occ_matrix_stride + lx == 511 ){ printf("\t\t 511: stride %d, lx %d, val %d\n",
                occ_matrix_stride,
                lx,
                n_local + offset + lx); }
                */
            }
        }
        return;
    }

}


int cudaHaloFillOccupancyMatrix(
    const int length,
    const int max_count,
    const int occ_matrix_stride,
    const int n_local,
    const int* __restrict__ d_halo_indices,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_halo_scan,
    int* __restrict__ d_occ_matrix
){
    dim3 bs, ts;
    cudaError_t err;

    /*
    int tmp;
    err=cudaMemcpy(&tmp, &d_occ_matrix[511], sizeof(int), cudaMemcpyDeviceToHost);
    cout << "err: " << err << endl;
    cout << "511 BEFORE OCC: " << tmp << endl;
    */



    err = cudaCreateLaunchArgs(length*max_count, 1024, &bs, &ts);
    if (err != cudaSuccess) { return err; }
    _cudaHaloFillOccupancyMatrix::fill_occ_matrix<<<bs,ts>>>(length*max_count,
                                                             max_count,
                                                             occ_matrix_stride,
                                                             n_local,
                                                             d_halo_indices,
                                                             d_ccc,
                                                             d_halo_scan,
                                                             d_occ_matrix);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return err;
        }


    /*
    err=cudaMemcpy(&tmp, &d_occ_matrix[511], sizeof(int), cudaMemcpyDeviceToHost);
    cout << "ptr in occ fill: " << (long)d_occ_matrix << endl;
    cout << "err: " << err << endl;
    cout << "511 AFTER OCC: " << tmp << endl;
    */


    return err;
}



/*
namespace _copysendcounts
{
    __global__ void _cudaCopySendCounts(
        const int * __restrict__ d_b_arr,
        const int * __restrict__ d_b_scan,
        int * __restrict__ d_p_count
    ){
        const int ix = threadIdx.x + blockIdx.x*blockDim.x;
        if (ix < 6){
            d_p_count[ix] = d_b_scan[d_b_arr[ix+1]] - d_b_scan[d_b_arr[ix]];
        }
        return;
    }
}
*/

int cudaCopySendCounts(
    const int * __restrict__ h_b_arr,
    const int * __restrict__ d_b_scan,
    int * __restrict__ h_p_count
){
    cudaError_t err;

    int p_count_tmp[6] = {0};

    err = cudaMemcpy(&p_count_tmp[0],
                     &d_b_scan[h_b_arr[1]],
                     sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return err; }


    err = cudaMemcpy(&p_count_tmp[1],
                     &d_b_scan[h_b_arr[2]],
                     sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return err; }


    err = cudaMemcpy(&p_count_tmp[2],
                     &d_b_scan[h_b_arr[3]],
                     sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return err; }


    err = cudaMemcpy(&p_count_tmp[3],
                     &d_b_scan[h_b_arr[4]],
                     sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return err; }


    err = cudaMemcpy(&p_count_tmp[4],
                     &d_b_scan[h_b_arr[5]],
                     sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return err; }



    err = cudaMemcpy(&p_count_tmp[5],
                     &d_b_scan[h_b_arr[6]],
                     sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return err; }


    h_p_count[0] = p_count_tmp[0];
    h_p_count[1] = p_count_tmp[1] - p_count_tmp[0];
    h_p_count[2] = p_count_tmp[2] - p_count_tmp[1];
    h_p_count[3] = p_count_tmp[3] - p_count_tmp[2];
    h_p_count[4] = p_count_tmp[4] - p_count_tmp[3];
    h_p_count[5] = p_count_tmp[5] - p_count_tmp[4];


    return 0;
}














































