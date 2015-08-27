#ifndef __CUDA_GENERIC__
#define __CUDA_GENERIC__

    #include "generic.h"
    #include <cuda.h>
    #include "helper_cuda.h"
    #include <vector_types.h>
    #include <cuda_profiler_api.h>



    // double shuffle down
    __device__ __inline__ double shfl_down_double(double x, int lane){

        int lo, hi;

        //split into parts
        asm volatile("mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

        // everyone do the shuffle. (32b registers)
        lo = __shfl_down(lo, lane);
        hi = __shfl_down(hi, lane);

        //recreate 64bits
        asm volatile( "mov.b64 %0, {%1,%2};" : "=d(x)" : "r"(lo) : "r"(hi));


        return x;
    }

    //double shuffle
    __device__ __inline__ double shfl_double(double x, int lane){

        int lo, hi;

        //split into parts
        asm volatile("mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

        // everyone do the shuffle. (32b registers)
        lo = __shfl(lo, lane);
        hi = __shfl(hi, lane);


        //recreate 64bits
        asm volatile( "mov.b64 %0, {%1,%2};" : "=d(x)" : "r"(lo) : "r"(hi));

        return x;
    }





#endif