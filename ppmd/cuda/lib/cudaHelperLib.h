#include <helper_cuda.h>


extern "C" int cudaErrorCheck(int err);
extern "C" int cudaCpyHostToDevice(void* dst, const void* src, size_t count);
extern "C" int cudaCpyDeviceToHost(void* dst, const void* src, size_t count);
extern "C" int cudaCpyDeviceToDevice(void* dst, const void* src, size_t count);
extern "C" int cudaHostRegisterWrapper(void* ptr, size_t size);
extern "C" int cudaHostUnregisterWrapper(void *ptr);
