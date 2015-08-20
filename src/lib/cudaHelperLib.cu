#include "cudaHelperLib.h"

int cudaErrorCheck(int err){
    checkCudaErrors((cudaError_t) err);
    return err;
}

void cudaCpyHostToDevice(void* dst, const void* src, size_t count){
    checkCudaErrors(cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}

void cudaCpyDeviceToHost(void* dst, const void* src, size_t count){
    checkCudaErrors(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}
