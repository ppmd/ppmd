
#include "cuda_generic.h"
#include <mpi.h>
#include <iostream>
#include <stdio.h>

using namespace std;




extern "C" int cudaNProcPBCStageOne(
    const int h_n,
    const double * __restrict__ h_B,
    double * __restrict__ d_P,
    const double * __restrict__ h_shifts,
    int * __restrict__ d_count,
    int * __restrict__ d_dir_count,
    int * __restrict__ d_escapees
);


extern "C" int cudaNProcPBCStageTwo(
    const int h_n,
    const int d_ncol,
    const int * __restrict__ d_escape_list,
    int * __restrict__ d_escape_matrix
);
