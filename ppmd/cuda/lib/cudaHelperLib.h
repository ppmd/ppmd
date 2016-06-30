#include <helper_cuda.h>
#include <iostream>
using namespace std;

extern "C" int cudaErrorCheck(int err);
extern "C" int cudaCpyHostToDevice(void* dst, const void* src, size_t count);
extern "C" int cudaCpyDeviceToHost(void* dst, const void* src, size_t count);
extern "C" int cudaCpyDeviceToDevice(void* dst, const void* src, size_t count);
extern "C" int cudaHostRegisterWrapper(void* ptr, size_t size);
extern "C" int cudaHostUnregisterWrapper(void *ptr);
extern "C" int cudaGetDeviceCountWrapper(int *count);
