
#include "cuda_generic.h"
#include <mpi.h>
#include <iostream>
#include <stdio.h>

using namespace std;



extern "C" int cudaHaloExchangePD(
    const int f_MPI_COMM,
    const int n_local,
    const int h_pos_flag,
    const int h_cccmax,
    const int h_occ_m_stride,
    const int* __restrict__ h_b_ind,
    const int* __restrict__ h_send_counts,
    const int* __restrict__ h_recv_counts,
    const int* __restrict__ SEND_RANKS,
    const int* __restrict__ RECV_RANKS,
    const int* __restrict__ d_b_indices,
    const int* __restrict__ d_occ_matrix,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_b_scan,
    const double* __restrict__ d_shift,
    %(DTYPE)s * __restrict__ d_ptr,
    %(DTYPE)s * __restrict__ d_buffer
);



