#ifndef __CUDASCAN__
#define __CUDASCAN__


#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "cuda_generic.h"

using namespace std;




namespace _thrust {
    template <typename T>
    int thrust_exclusive_scan(T* d_ptr, const int len){

        thrust::device_ptr<T> td_ptr = thrust::device_pointer_cast(d_ptr);
        thrust::exclusive_scan(td_ptr, td_ptr + len, td_ptr);

        return 0;
    }
}


extern "C" int cudaExclusiveScanDouble(double * d_ptr, const int len);
extern "C" int cudaExclusiveScanInt(int * d_ptr, const int len);












#endif