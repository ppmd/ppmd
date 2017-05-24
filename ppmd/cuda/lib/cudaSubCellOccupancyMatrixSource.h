#include <cuda_generic.h>

extern "C" int LayerSort(
    const int blocksize[3],
    const int threadsize[3],
    const int blocksize2[3],
    const int threadsize2[3],
    const int n,
    const int nc,
    int* nl,
    int* __restrict__ d_pl,
    int* __restrict__ d_crl,
    int* __restrict__ d_ccc,
    const int* __restrict__ h_ca,
    const double* __restrict__ h_b,
    const double* __restrict__ h_cel,
    const double* __restrict__ d_p
);

extern "C" int PopMatrix(
    const int blocksize[3],
    const int threadsize[3],
    const int n,
    const int nl,
    const int* __restrict__ d_pl,
    const int* __restrict__ d_crl,
    int* __restrict__ d_M
);

