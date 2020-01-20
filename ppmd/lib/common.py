from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import cgen
import ctypes


allowed_dtypes = (
    ctypes.c_double,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_int64,
    ctypes.c_uint64,
    ctypes.c_byte,
    'float64',
    'int32'
)


_ctypes_map = {
    ctypes.c_int64: 'int64_t',
    ctypes.c_uint64: 'uint64_t',
    ctypes.c_int32: 'int32_t',
    ctypes.c_uint32: 'uint32_t'   
}


class DtypeToCtype:
    def __init__(self, existing):
        self._e = existing

    def __call__(self, key):
        if key in self._e.keys():
            return self._e[key]
        else:
            return cgen.dtype_to_ctype(key)

    def __getitem__(self, key):
        return self(key)


ctypes_map = DtypeToCtype(_ctypes_map)

OMP_DECOMP_HEADER=r'''

#include <stdlib.h>

#ifndef TESTING
#include <omp.h>
#endif

#ifndef TESTING
#define get_num_threads omp_get_num_threads
#else
#define get_num_threads test_get_num_threads
#endif

#ifndef TESTING
#define get_thread_num omp_get_thread_num
#else
#define get_thread_num test_get_thread_num
#endif

#ifdef TESTING
static int num_threads = 0;
static int thread_num = 0;

static int test_get_num_threads(){return num_threads;}
static int test_get_thread_num(){return thread_num;}

extern "C"
int test_set_num_threads(const int n){num_threads = n; return 0;}

extern "C"
int test_set_thread_num(const int n){thread_num = n; return 0;}
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef TESTING
extern "C"
#else
static inline
#endif
int get_thread_decomp(const int N, int * rstart, int * rend){
    
    const div_t pq = div(N, get_num_threads());
    const int i = get_thread_num();
    const int p = pq.quot;
    const int q = pq.rem;
    const int n = (i < q) ? (p + 1) : p;
    const int start = (MIN(i, q) * (p + 1)) + ((i > q) ? (i - q) * p : 0);
    const int end = start + n;
    
    *rstart = start;
    *rend = end;
    
    return 0;
}

'''
