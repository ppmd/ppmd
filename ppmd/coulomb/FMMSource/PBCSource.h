#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define REAL double
#define INT64 int64_t

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









