#ifndef __COUNTING_TYPES__
#define __COUNTING_TYPES__

#include "cuda_generic.h"

#define _ulong unsigned long long int




namespace cuda_double_counters{
    _ulong *a, *s, *m, *d, *r, *w;

    __constant__ _ulong *d_a, *d_s, *d_m, *d_d, *d_r, *d_w; 

    _ulong h_a = 0, h_s =0, h_m = 0, h_d = 0, h_r = 0, h_w = 0; 

    std::size_t size = sizeof(double);

    void Init();
    void Finalise();
    
}

void cuda_double_counters::Init(){
    checkCudaErrors(cudaMalloc(&a, sizeof(_ulong)));
    checkCudaErrors(cudaMalloc(&s, sizeof(_ulong)));   
    checkCudaErrors(cudaMalloc(&m, sizeof(_ulong)));
    checkCudaErrors(cudaMalloc(&d, sizeof(_ulong)));
    checkCudaErrors(cudaMalloc(&r, sizeof(_ulong)));
    checkCudaErrors(cudaMalloc(&w, sizeof(_ulong)));

    checkCudaErrors(cudaMemcpyToSymbol(d_a, &a, sizeof(_ulong*)));
    checkCudaErrors(cudaMemcpyToSymbol(d_s, &s, sizeof(_ulong*)));
    checkCudaErrors(cudaMemcpyToSymbol(d_m, &m, sizeof(_ulong*)));
    checkCudaErrors(cudaMemcpyToSymbol(d_d, &d, sizeof(_ulong*)));
    checkCudaErrors(cudaMemcpyToSymbol(d_r, &r, sizeof(_ulong*)));
    checkCudaErrors(cudaMemcpyToSymbol(d_w, &w, sizeof(_ulong*)));


    checkCudaErrors(cudaMemset(a, 0, sizeof(_ulong)));
    checkCudaErrors(cudaMemset(s, 0, sizeof(_ulong)));
    checkCudaErrors(cudaMemset(m, 0, sizeof(_ulong)));
    checkCudaErrors(cudaMemset(d, 0, sizeof(_ulong)));
    checkCudaErrors(cudaMemset(r, 0, sizeof(_ulong)));
    checkCudaErrors(cudaMemset(w, 0, sizeof(_ulong)));
}

void cuda_double_counters::Finalise(){
    checkCudaErrors(cudaMemcpy(&h_a, a, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_s, s, sizeof(_ulong), cudaMemcpyDeviceToHost));   
    checkCudaErrors(cudaMemcpy(&h_m, m, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_d, d, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_r, r, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_w, w, sizeof(_ulong), cudaMemcpyDeviceToHost));
}






class cuda_double_prof {

    public:
        double v; // This is the only data storage 
                  // otherwise there will be a size missmatch.
        
        __host__ __device__ cuda_double_prof(){ this->v = 0; }
        
        //Construct from some type
        template <typename T>
        __host__ __device__ cuda_double_prof(T v){ this->v = (double) v; }

        //Cast to some type
        template <typename T>
        __host__ __device__ operator T() {return ((T) this->v);}

        //operators.
        __device__ cuda_double_prof operator+(const cuda_double_prof& other) const;
        __device__ cuda_double_prof operator+=(const cuda_double_prof& other) const;
        __device__ cuda_double_prof operator*(const cuda_double_prof& other) const;
        __device__ cuda_double_prof operator/(const cuda_double_prof& other) const;
        __device__ cuda_double_prof operator-(const cuda_double_prof& other) const;

};

__device__ cuda_double_prof cuda_double_prof::operator+(const cuda_double_prof& other) const {
    atomicAdd(cuda_double_counters::d_a, (_ulong)1);
    return cuda_double_prof(this->v + other.v);
} 

__device__ cuda_double_prof cuda_double_prof::operator+=(const cuda_double_prof& other) const {
    atomicAdd(cuda_double_counters::d_a, (_ulong)1);
    
    return cuda_double_prof(this->v + other.v);
} 

__device__ cuda_double_prof cuda_double_prof::operator*(const cuda_double_prof& other) const {
    atomicAdd(cuda_double_counters::d_m, (_ulong)1);
    return cuda_double_prof(this->v * other.v);
} 

__device__ cuda_double_prof cuda_double_prof::operator/(const cuda_double_prof& other) const {
    atomicAdd(cuda_double_counters::d_d, (_ulong)1);
    return cuda_double_prof(this->v / other.v);
}

__device__ cuda_double_prof cuda_double_prof::operator-(const cuda_double_prof& other) const {
    atomicAdd(cuda_double_counters::d_s, (_ulong)1);
    return cuda_double_prof(this->v - other.v);
}




/*
namespace cuda_int_counters{
    ulong a = 0;    // addition
    ulong s = 0;    // subtraction
    ulong m = 0;    // multiplication
    ulong d = 0;    // division
    ulong r = 0;    // reads
    ulong w = 0;    // writes
    std::size_t size = sizeof(int);
}


class cuda_int_prof {

    public:
        int v; // This is the only data storage 
                  // otherwise there will be a size missmatch.
        
        cuda_int_prof(){ this->v = 0; }
        cuda_int_prof(int v){ this->v = v; }
        

        //operators.
        cuda_int_prof operator+(const cuda_int_prof& other);
        cuda_int_prof operator+=(const cuda_int_prof& other);
        cuda_int_prof operator*(const cuda_int_prof& other);
        cuda_int_prof operator/(const cuda_int_prof& other);
        cuda_int_prof operator-(const cuda_int_prof& other);

};

cuda_int_prof cuda_int_prof::operator+(const cuda_int_prof& other){
    cuda_int_counters::a ++;
    return cuda_int_prof(this->v + other.v);
}

cuda_int_prof cuda_int_prof::operator+=(const cuda_int_prof& other){
    cuda_int_counters::a ++;
    return cuda_int_prof(this->v += other.v);
}

cuda_int_prof cuda_int_prof::operator*(const cuda_int_prof& other){
    cuda_int_counters::m ++;
    return cuda_int_prof(this->v * other.v);
}

cuda_int_prof cuda_int_prof::operator/(const cuda_int_prof& other){
    cuda_int_counters::d ++;
    return cuda_int_prof(this->v / other.v);
}

cuda_int_prof cuda_int_prof::operator-(const cuda_int_prof& other){
    cuda_int_counters::s ++;
    return cuda_int_prof(this->v - other.v);
}
*/


#endif
