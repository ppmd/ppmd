/*

Certain portions of this file are copied/adapted from:

https://github.com/parallel-forall/code-samples

And fall under a BSD license. These lines of code are found in the later
lines of this file with a copy of the license included.

*/

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



    //using namespace std;


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
        std::cout << "cudaCreateLaunchArgs Error: Invalid desired number of total threads " << N <<
            "or invalid number of threads per block " << Nt << std::endl;
        return cudaErrorUnknown;
    }

    const int Nb = ceil(((double) N) / ((double) Nt));

    bs->x = Nb; bs->y = 1; bs->z = 1;
    ts->x = Nt; ts->y = 1; ts->z = 1;

    return cudaSuccess;
}


/*__device__ bool isnormal(double value)
{
    return !(isinf(value) || isnan(value));
}
*/




/*
Lines below this point are subject to the following license.


# Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



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

    // Taken from http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    __inline__ __device__
    int warpReduceSumInt(int val) {
      return warpReduceSum(val);
    }

    __inline__ __device__
    int atomicAddInt(int* address, int val){
        return atomicAdd(address, val);
    }




    // edited from http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    __inline__ __device__
    double warpReduceSumDouble(double val) {
      for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += shfl_down_double(val, offset);
      return val;
    }

    __inline__ __device__
    double warpReduceMaxDouble(double val) {
      for (int offset = warpSize/2; offset > 0; offset /= 2)
        val = max(shfl_down_double(val, offset), val);
      return val;
    }

#if __CUDA_ARCH__ < 350

    __inline__ __device__ double atomicMaxDouble(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(
                                max(val, __longlong_as_double(assumed))
                            )
            );

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }

#else

    __inline__ __device__ double atomicMaxDouble(double *address, double val){
        union
        {
            signed long long int dint;
            double d;
        } uval_t;

        uval_t.d = val;
        uval_t.dint = atomicMax((signed long long int*) address, uval_t.dint);

        return uval_t.d;
    }


#endif

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







#endif
