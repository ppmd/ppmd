#include "cudaMPILib.h"



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


    __global__ void cudaFindNewSlots_kernel2(const int * __restrict__ d_scan,
                                             int * __restrict__ d_sources){

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





cudaError_t cudaCreateLaunchArgs(
        const int N,    // Total minimum number of threads.
        const int Nt,   // Number of threads per block.
        dim3* bs,       // RETURN: grid of thread blocks.
        dim3* ts        // RETURN: grid of threads
        ){

    if ((N<0) || (Nt<0)){
        cout << "cudaCreateLaunchArgs Error: Invalid desired number of total threads " << N << 
            "or invalid number of threads per block " << Nt << endl;
        return cudaErrorUnknown;
    }
    
    const int Nb = ceil(((double) N) / ((double) Nt));

    bs->x = Nb; bs->y = 1; bs->z = 1;
    ts->x = Nt; ts->y = 1; ts->z = 1;

    return cudaSuccess;
}





/*
namespace _ExSizes
{   

    __global__ void GatherCellCounts(
        const int n,                        // Number of cells to inspect
        const int* __restrict__ D_b_arr,    // actual indices of boundary cells
        int* D_b_tmp,                       // space to place boundary cell counts
        int* D_tmp_count                    // reduce the count accross cells into here
        ){


        int _ix = threadIdx.x + blockIdx.x*blockDim.x;
        int tt = 0;


        if (_ix < n){

            D_b_tmp[_ix] = D_b_arr[_ix];   // copy this cell count
            tt = D_b_arr[_ix];             // set local tt to this cell count

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
        const int* __restrict__ D_CCC,          // Cell contents count array
        int* __restrict__ H_halo_count,         // RETURN: Number of halo particles
        int* __restrict__ H_tmp_count,          // RETURN: Amount of temporary space needed
        int* __restrict__ D_h_tmp,              // Temp storage for halo counts
        int* __restrict__ H_b_tmp,              // Temp storage for bundary counts
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

    int * D_tmp_count;
    
    // make a device tmp
    err = (int) cudaMalloc(&D_tmp_count, sizeof(int)); 
    if (err != 0) { return err; }
    
    // ensure is zero
    err = (int) cudaMemcpy(D_tmp_count, H_tmp_count, sizeof(int), cudaMemcpyHostToDevice);
    if (err != 0) { return err; }



    for(int dir=0 ; dir<6 ; dir++ ){

        // Here we want to collect the local cell counts for a direction on the device
        // exchange these sizes and get the total for the direction
    
        const int dir_cell_count = H_b_ind[dir+1] - H_b_ind[dir];
        err = cudaCreateLaunchArgs(dir_cell_count, 256, &bs, &ts);
        if (err != 0) { return err; }
        
        _ExSizes::GatherCellCounts<<<bs,ts>>>(dir_cell_count,
                                              D_b_arr+H_b_ind[dir],
                                              D_b_tmp,
                                              D_tmp_count);
        
        err = cudaDeviceSynchronize();
        if (err != 0) { return err; }

    }


    return 0;
}

*/

















