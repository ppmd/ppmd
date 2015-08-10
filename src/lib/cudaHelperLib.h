#include <helper_cuda.h>


extern "C" int cudaErrorCheck(int err);
extern "C" void cudaCpyHostToDevice(void* dst, const void* src, size_t count);
extern "C" void cudaCpyDeviceToHost(void* dst, const void* src, size_t count);
