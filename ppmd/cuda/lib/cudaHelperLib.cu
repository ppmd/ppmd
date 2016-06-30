#include "cudaHelperLib.h"
#include <iostream>

int cudaErrorCheck(int err){
    //checkCudaErrors((cudaError_t) err);
    if (err != 0) {
        std::cout << cudaGetErrorString((cudaError_t) err) << std::endl;
    }
    return err;
}

int cudaCpyHostToDevice(void* dst, const void* src, size_t count){
    cudaError_t err;
    err = cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice);
    return (int) err;
}

int cudaCpyDeviceToHost(void* dst, const void* src, size_t count){
    cudaError_t err;
    err = cudaMemcpy(dst,
                src,
                count,
                cudaMemcpyDeviceToHost);
    return (int) err;
}

int cudaCpyDeviceToDevice(void* dst, const void* src, size_t count){
    cudaError_t err;
    err = cudaMemcpy(dst,src,count,cudaMemcpyDeviceToDevice);
    return (int) err;
}

int cudaHostRegisterWrapper(void* ptr, size_t size){
    cudaError_t err;
    err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    return (int) err;
}

int cudaHostUnregisterWrapper(void* ptr){
    cudaError_t err;
    err = cudaHostUnregister(ptr);
    return (int) err;
}
