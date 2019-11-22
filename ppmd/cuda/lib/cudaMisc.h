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
    cudaError_t check_pointer_is_device(const T * d_ptr){
        
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, d_ptr);
        if (err != cudaSuccess) { return err; }
        
        // if pointer cannot be dereferenced on device
        if (attributes.devicePointer == NULL) {return cudaErrorInvalidDevicePointer;}

        return err;
    }

    template <typename T>
    cudaError_t thrust_max_element(T* d_ptr, const int len, T * max_element){
        
        cudaError_t err = check_pointer_is_device<T>(d_ptr);
        if (err != cudaSuccess) { return err; }

        thrust::device_ptr<T> td_ptr_start = thrust::device_pointer_cast(d_ptr);
        *max_element = *(thrust::max_element(td_ptr_start, td_ptr_start + len));

        return cudaSuccess;
    }

    template <typename T>
    cudaError_t thrust_min_element(T* d_ptr, const int len, T * min_element){

        cudaError_t err = check_pointer_is_device<T>(d_ptr);
        if (err != cudaSuccess) { return err; }

        thrust::device_ptr<T> td_ptr = thrust::device_pointer_cast(d_ptr);
        *min_element = *(thrust::min_element(td_ptr, td_ptr + len));
        
        return cudaSuccess;
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

extern "C" cudaError_t cudaMaxElementInt(int * d_ptr, const int len, int * max_element){
    return _thrust::thrust_max_element<int>(d_ptr, len, max_element);
}


extern "C" cudaError_t cudaMinElementInt(int * d_ptr, const int len, int * min_element){
    return _thrust::thrust_min_element<int>(d_ptr, len, min_element);
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
