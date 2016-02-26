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
    checkCudaErrors(cudaMemcpy(dst,
                src,
                count,
                cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}

void cudaCpyDeviceToDevice(void* dst, const void* src, size_t count){
    checkCudaErrors(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}

void cudaHostRegisterWrapper(void* ptr, size_t size){
    checkCudaErrors(cudaHostRegister(ptr, size, cudaHostRegisterPortable));
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}

void cudaHostUnregisterWrapper(void* ptr){
    checkCudaErrors(cudaHostUnregister(ptr));
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}
