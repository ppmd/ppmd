
//#define USE_MKL
#ifdef USE_MKL
    #include "mkl.h"
#endif

#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define REAL double
#define INT64 int64_t
#define INT32 int32_t

using namespace std;


#define _LM_TO_IND(L, M) ((L)+(M))

#define EXP_RE_IND(L, M) (_LM_TO_IND((L), (M)))
#define EXP_IM_IND(L, M) ((2*(L))+1 + EXP_RE_IND((L),(M)))

#define CUBE_IND(L, M) ((L) * ( (L) + 1 ) + (M) )

// defined for non-negative M
#define P_IND(L, M) (((L)*((L) + 1)/2) + (M))

#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x < y) ? y : x)

#define I_IND(nlevel, kx, mx) ((2*(nlevel)+1)*(nlevel+(kx)) + nlevel + (mx) )

#define ABS(x) ((x) > 0 ? (x) : -1*(x))

#define BLOCK_SIZE %(SUB_BLOCK_SIZE)s
#define ASTRIDE1 %(SUB_ASTRIDE1)s
#define ASTRIDE2 %(SUB_ASTRIDE2)s
#define DGEMM %(SUB_DGEMM)s

%(SUB_IARRAY)s

#define DOUBLE double
#define SINGLE float





/*
Layout of memory used by above macros.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| -L | - - - - | -1 | 0 | 1 | - - - - | L |

and

| Re coeffs for all l | Im coeffs for all l |

*/






