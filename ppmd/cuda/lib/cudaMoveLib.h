
#include "cuda_generic.h"
#include <mpi.h>
#include <iostream>
#include <stdio.h>

using namespace std;




extern "C" int cudaMoveStageOne(
    const int FCOMM,
    const int * h_send_ranks,
    const int * h_recv_rank,
    int * h_send_counts,
    int * h_recv_counts
);




extern "C" int cudaMoveStageTwo(
    const int FCOMM,
    const int n_local,
    const int total_bytes,
    const int num_dats,
    const int * __restrict__ h_send_counts,
    const int * __restrict__ h_recv_counts,
    const int * __restrict__ h_send_ranks,
    const int * __restrict__ h_recv_ranks,
    const int * __restrict__ d_move_matrix,
    const int move_matrix_stride,
    char * __restrict__ d_send_buf,
    char * __restrict__ d_recv_buf,
    void ** h_ptrs,
    const int * __restrict__ h_byte_counts,
    int * __restrict__ d_empty_flag
);
