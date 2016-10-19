

#include "cuda_generic.h"
#include <mpi.h>
#include <iostream>
#include <stdio.h>

using namespace std;

#define TYPE %(TYPENAME)s

#define _ABS(x) (((x)>0)?(x):(  ((TYPE) -1) * (x)  ))


extern "C" int cudaLInfNorm(
    const TYPE * __restrict__ d_ptr,
    const int len,
    double *val
);
