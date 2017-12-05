
#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>

#define REAL double
#define UINT64 uint64_t
#define INT64 int64_t
#define UINT32 int32_t
#define INT32 int32_t

using namespace std;

#define _LM_TO_IND(L, M) ((L)+(M))

#define EXP_RE_IND(L, M) (_LM_TO_IND((L), (M)))
#define EXP_IM_IND(L, M) ((2*(L))+1 + EXP_RE_IND((L),(M)))

#define CUBE_IND(L, M) ((L) * ( (L) + 1 ) + (M) )

#define P_SPACE_IND(LMAX, L, M) (((LMAX)+1)*(L) + (M))


/*
Layout of memory used by above macros.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| -L | - - - - | -1 | 0 | 1 | - - - - | L |

and

| Re coeffs for all l | Im coeffs for all l |

*/



