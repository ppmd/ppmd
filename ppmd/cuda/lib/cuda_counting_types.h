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
        __host__ __device__ operator T() const {return ((T) this->v);}

        //maths operators.
        template <typename T>
        __device__ T operator+(const T& other) const;
        template <typename T>
        __device__ T operator+=(const T& other) const;
        template <typename T>
        __device__ T operator*(const T& other) const;
        template <typename T>
        __device__ T operator/(const T& other) const;
        template <typename T>
        __device__ T operator-(const T& other) const;

        //other operators
        __device__ double* operator&() const;

};

template <typename T>
__device__ T cuda_double_prof::operator+(const T& other) const {
    atomicAdd(cuda_double_counters::d_a, (_ulong)1);
    return cuda_double_prof(this->v + (double)other);
} 

template <typename T>
__device__ T cuda_double_prof::operator+=(const T& other) const {
    atomicAdd(cuda_double_counters::d_a, (_ulong)1);
    return cuda_double_prof(this->v + (double)other);
} 

template <typename T>
__device__ T cuda_double_prof::operator*(const T& other) const {
    atomicAdd(cuda_double_counters::d_m, (_ulong)1);
    return cuda_double_prof(this->v * (double)other);
} 

template <typename T>
__device__ T cuda_double_prof::operator/(const T& other) const {
    atomicAdd(cuda_double_counters::d_d, (_ulong)1);
    return cuda_double_prof(this->v / (double)other);
}

template <typename T>
__device__ T cuda_double_prof::operator-(const T& other) const {
    atomicAdd(cuda_double_counters::d_s, (_ulong)1);
    return cuda_double_prof(this->v - (double)other);
}



__device__ double* cuda_double_prof::operator&() const{
    return (double*)&v;
}











// ----------------------------------------------------------------------


namespace cuda_int_counters{
    _ulong *a, *s, *m, *d, *r, *w;

    __constant__ _ulong *d_a, *d_s, *d_m, *d_d, *d_r, *d_w; 

    _ulong h_a = 0, h_s =0, h_m = 0, h_d = 0, h_r = 0, h_w = 0; 

    std::size_t size = sizeof(int);

    void Init();
    void Finalise();
    
}

void cuda_int_counters::Init(){
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

void cuda_int_counters::Finalise(){
    checkCudaErrors(cudaMemcpy(&h_a, a, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_s, s, sizeof(_ulong), cudaMemcpyDeviceToHost));   
    checkCudaErrors(cudaMemcpy(&h_m, m, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_d, d, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_r, r, sizeof(_ulong), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_w, w, sizeof(_ulong), cudaMemcpyDeviceToHost));
}


class cuda_int_prof {

    public:
        int v; // This is the only data storage 
                  // otherwise there will be a size missmatch.
        
        __host__ __device__ cuda_int_prof(){ this->v = 0; }
        
        //Construct from some type
        template <typename T>
        __host__ __device__ cuda_int_prof(T v){ this->v = (int) v; }

        //Cast to some type
        template <typename T>
        __host__ __device__ operator T() const {return ((T) this->v);}

        // operators.
        template <typename T>
        __device__ T operator+(const T& other) const;
        template <typename T>
        __device__ T operator+=(const T& other) const;
        template <typename T>
        __device__ T operator*(const T& other) const;
        template <typename T>
        __device__ T operator/(const T& other) const;
        template <typename T>
        __device__ T operator-(const T& other) const;

        __device__ int* operator& () const;

};

template <typename T>
__device__ T cuda_int_prof::operator+(const T& other) const {
    atomicAdd(cuda_int_counters::d_a, (_ulong)1);
    return cuda_int_prof(this->v + (int)other);
} 

template <typename T>
__device__ T cuda_int_prof::operator+=(const T& other) const {
    atomicAdd(cuda_int_counters::d_a, (_ulong)1);
    return cuda_int_prof(this->v + (int)other);
} 

template <typename T>
__device__ T cuda_int_prof::operator*(const T& other) const {
    atomicAdd(cuda_int_counters::d_m, (_ulong)1);
    return cuda_int_prof(this->v * (int)other);
} 

template <typename T>
__device__ T cuda_int_prof::operator/(const T& other) const {
    atomicAdd(cuda_int_counters::d_d, (_ulong)1);
    return cuda_int_prof(this->v / (int)other);
}

template <typename T>
__device__ T cuda_int_prof::operator-(const T& other) const {
    atomicAdd(cuda_int_counters::d_s, (_ulong)1);
    return cuda_int_prof(this->v - (int)other);
}



__device__ int* cuda_int_prof::operator&() const{
    return (int*) &v;
}


#endif
