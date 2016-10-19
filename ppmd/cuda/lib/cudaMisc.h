#ifndef __CUDASCAN__
#define __CUDASCAN__


#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include "cuda_generic.h"

using namespace std;


namespace _thrust {
    template <typename T>
    int thrust_exclusive_scan(T* d_ptr, const int len){

        thrust::device_ptr<T> td_ptr = thrust::device_pointer_cast(d_ptr);
        thrust::exclusive_scan(td_ptr, td_ptr + len, td_ptr);

        return 0;
    }

    template <typename T>
    T thrust_max_element(T* d_ptr, const int len){

        thrust::device_ptr<T> td_ptr = thrust::device_pointer_cast(d_ptr);
        return *(thrust::max_element(td_ptr, td_ptr + len));
    }

    template <typename T>
    T thrust_min_element(T* d_ptr, const int len){

        thrust::device_ptr<T> td_ptr = thrust::device_pointer_cast(d_ptr);
        return *(thrust::min_element(td_ptr, td_ptr + len));
    }


}


extern "C" int cudaExclusiveScanDouble(double * d_ptr, const int len){
    _thrust::thrust_exclusive_scan<double>(d_ptr, len);
    return 0;
}

extern "C" int cudaExclusiveScanInt(int * d_ptr, const int len){
    _thrust::thrust_exclusive_scan<int>(d_ptr, len);
    return 0;
}

extern "C" int cudaMaxElementInt(int * d_ptr, const int len){
    return _thrust::thrust_max_element<int>(d_ptr, len);
}
extern "C" int cudaMinElementInt(int * d_ptr, const int len){
    return _thrust::thrust_min_element<int>(d_ptr, len);
}





extern "C" int cudaMemSetZero(
     	void *devPtr,
		int value,
		size_t count){
    return cudaMemset(
     	devPtr,
		value,
		count
	);
}











#endif