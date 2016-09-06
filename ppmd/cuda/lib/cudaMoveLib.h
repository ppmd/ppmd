
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
