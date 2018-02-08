#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define REAL double
#define UINT64 uint64_t
#define INT64 int64_t
#define UINT32 int32_t
#define INT32 int32_t

using namespace std;

struct CART {
    REAL x;
    REAL y;
    REAL z;
};

struct SPH {
    REAL radius;
    REAL phi;
    REAL theta;
};









