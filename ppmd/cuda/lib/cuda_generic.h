#ifndef __CUDA_GENERIC__
#define __CUDA_GENERIC__

    #include "generic.h"
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include "helper_cuda.h"
    #include <builtin_types.h>
    #include <cuda_profiler_api.h>
    #include <device_functions.h>
    //#include "cuda_counting_types.h"
    #include <iostream>

__device__ bool isnormal(double value)
{
	return !(isinf(value) || isnan(value));
}
    using namespace std;
    /*
    // double shuffle down edited from nvidia example
    __device__ __inline__ double shfl_down_double(double x, int lane){

        int lo, hi;

        //split into parts
        asm volatile("mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

        // everyone do the shuffle. (32b registers)
        lo = __shfl_down(lo, lane);
        hi = __shfl_down(hi, lane);

        //recreate 64bits
        asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo) : "r"(hi) );


        return x;
    }

    //double shuffle edited from nvidia example
    __device__ __inline__ double shfl_double(double x, int lane){

        int lo, hi;

        //split into parts
        asm volatile("mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

        // everyone do the shuffle. (32b registers)
        lo = __shfl(lo, lane);
        hi = __shfl(hi, lane);


        //recreate 64bits
        asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo) : "r"(hi) );

        return x;
    }
    */
    
    __device__ inline
    double shfl_down_double(double var, unsigned int srcLane, int width=32) {
      int2 a = *reinterpret_cast<int2*>(&var);
      a.x = __shfl_down(a.x, srcLane, width);
      a.y = __shfl_down(a.y, srcLane, width);
      return *reinterpret_cast<double*>(&a);
    }    
    

    // Taken from http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    __inline__ __device__
    int warpReduceSum(int val) {
      for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
      return val;
    }

    // edited from http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    __inline__ __device__
    double warpReduceSumDouble(double val) {
      for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += shfl_down_double(val, offset);
      return val;
    }

    // Taken from http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    __inline__ __device__
    int warpReduceMax(int val) {
      for (int offset = warpSize/2; offset > 0; offset /= 2)
        val = max(__shfl_down(val, offset), val);
      return val;
    }

    // atomic addition edited from Jon Cohen (NVIDIA)
    __device__ static double atomicAddDouble(double *addr, double val){
        double old=*addr, assumed;

        do {

            assumed = old;
            old = __longlong_as_double(
            atomicCAS((unsigned long long int*)addr,
              __double_as_longlong(assumed),
              __double_as_longlong(val+assumed) )
            );

        } while (assumed!=old);

        return old;
    }


template <typename T>
struct cuda_Array {
    T* __restrict__ ptr;
    int *ncomp;
};

template <typename T>
struct cuda_Matrix {
    T* __restrict__ ptr;
    int *nrow;
    int *ncol;
};

template <typename T>
struct cuda_ParticleDat {
    T* __restrict__ ptr;
    int* nrow;
    int* ncol;
    int* npart;
    int* ncomp;
};


template <typename T>
struct const_cuda_ParticleDat {
    const T* __restrict__ ptr;
    const int* nrow;
    const int* ncol;
    const int* npart;
    const int* ncomp;
};






cudaError_t cudaCreateLaunchArgs(
        const int N,    // Total minimum number of threads.
        const int Nt,   // Number of threads per block.
        dim3* bs,       // RETURN: grid of thread blocks.
        dim3* ts        // RETURN: grid of threads
        ){

    if ((N<0) || (Nt<0)){
        cout << "cudaCreateLaunchArgs Error: Invalid desired number of total threads " << N <<
            "or invalid number of threads per block " << Nt << endl;
        return cudaErrorUnknown;
    }

    const int Nb = ceil(((double) N) / ((double) Nt));

    bs->x = Nb; bs->y = 1; bs->z = 1;
    ts->x = Nt; ts->y = 1; ts->z = 1;

    return cudaSuccess;
}






#endif
