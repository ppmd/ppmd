






__global__ void cudaLInfNorm_k(
    const TYPE * __restrict__ d_ptr,
    const int len,
    double * __restrict__ d_val
){

    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    double t = 0.0;
    if (ix<len){
        t = (double) _ABS(d_ptr[ix]);
    }

    t = warpReduceMaxDouble(t);

    __shared__ double dt[1];

    if (  (int)(threadIdx.x & (warpSize - 1)) == 0){
      dt[0] = 0;
    }

    __syncthreads();


    if (  (int)(threadIdx.x & (warpSize - 1)) == 0){
        atomicMaxDouble(&dt[0], t);
    }
    __syncthreads();

    if (threadIdx.x == 0){
         atomicMaxDouble(&d_val[0], dt[0]);
    }

    return;
}


int cudaLInfNorm(
    const TYPE * __restrict__ d_ptr,
    const int len,
    double *val
)
{
    cudaError_t err;
    dim3 bs, ts;
    *val = 0;

    double *d_val;
    err = cudaMalloc(&d_val, sizeof(double));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpy(d_val, val, sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {return err;}

    err = cudaCreateLaunchArgs(len, 512, &bs, &ts);
    if (err != cudaSuccess) { return err; }

    cudaLInfNorm_k<<<bs, ts>>>(
        d_ptr,
        len,
        d_val
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return err;}

    return cudaMemcpy(val, d_val, sizeof(double), cudaMemcpyDeviceToHost);
}



