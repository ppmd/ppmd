#include <omp.h>
#include <stdio.h>

const int h_map[27][3] = {
        {-1,1,-1},
        {-1,-1,-1},
        {-1,0,-1},
        {0,1,-1},
        {0,-1,-1},
        {0,0,-1},
        {1,0,-1},
        {1,1,-1},
        {1,-1,-1},

        {-1,1,0},
        {-1,0,0},
        {-1,-1,0},
        {0,-1,0},
        {0,0,0},
        {0,1,0},
        {1,0,0},
        {1,1,0},
        {1,-1,0},

        {-1,0,1},
        {-1,1,1},
        {-1,-1,1},
        {0,0,1},
        {0,1,1},
        {0,-1,1},
        {1,0,1},
        {1,1,1},
        {1,-1,1}
};
extern "C"
int OMPNeighbourMatrix(
        const int NPART,
        const double *P,
        const int *qlist,
        const int qoffset,
        const int *CRL,
        const int *CA,
        int *NLIST,
        int *NNEIG,
        const int STRIDE,
        const double cutoff
);