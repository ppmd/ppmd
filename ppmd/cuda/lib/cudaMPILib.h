#include <helper_cuda.h>
#include <iostream>
#include <mpi.h>
#include "cuda_generic.h"
#include "cudaMisc.h"
#include <stdio.h>

using namespace std;


// Directly related to MPI
extern "C" int MPIErrorCheck_cuda(const int error_code);
extern "C" int MPI_Bcast_cuda(const int FCOMM,
                              void* buffer,
                              const int byte_count,
                              const int root
                              );


extern "C" int MPI_Gatherv_cuda(const int FCOMM,
                                const void* s_buffer,
                                const int s_count,
                                void* r_buffer,
                                const int* r_counts,
                                const int* r_disps,
                                const int root
                                );










// MPI related multigpu static libs
extern "C" int cudaFindEmptySlots(const int blocksize1[3],
                                  const int threadsize1[3],
                                  const int* d_scan,
                                  const int h_n1,
                                  int * d_empties
                                  );

extern "C" int cudaFindNewSlots(const int blocksize2[3],
                                const int threadsize2[3],
                                const int* d_scan,
                                const int h_n1,
                                const int h_n2,
                                int * d_sources
                                );

extern "C" int cudaHaloArrayCopyScan(const int length,
                                     const int* __restrict__ d_map,
                                     const int* __restrict__ d_ccc,
                                     int* __restrict__ d_scan,
                                     int* __restrict__ h_max
                                     );

extern "C" int cudaHaloFillOccupancyMatrix(
    const int length,
    const int max_count,
    const int occ_matrix_stride,
    const int n_local,
    const int* __restrict__ d_halo_indices,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_halo_scan,
    int* __restrict__ d_occ_matrix
);



extern "C" int cudaCopySendCounts(
    const int * __restrict__ h_b_arr,
    const int * __restrict__ d_b_scan,
    int * __restrict__ h_p_count
);





// CUDA MPI exchange sizes functions
// Host pointers begin with H_, device pointers begin with D_.
// Essentially follows the host code in a highly multithreaded way.
/*
extern "C" int cudaExchangeCellCounts(
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
        );
*/











