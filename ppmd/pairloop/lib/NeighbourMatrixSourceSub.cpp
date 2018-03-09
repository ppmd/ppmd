

extern "C"
int OMPNeighbourMatrixSub(
        const INT64 NPART,
        const REAL * RESTRICT P,
        const INT64 *qlist,
        const INT64 qoffset,
        const INT64 *CRL,
        const INT64 *CA,
        INT64 * RESTRICT NLIST,
        INT64 * RESTRICT NNEIG,
        const INT64 STRIDE,
        const REAL cutoff,
        INT64 * RESTRICT total_num_neigh
){
    INT64 totaln = 0;
    int err = 0;
    INT64 tmp_offset[27];
    for(INT64 ix=0; ix<27; ix++){
        tmp_offset[ix] = h_map[ix][0] + h_map[ix][1] * CA[0] + h_map[ix][2] * CA[0]* CA[1];
    }
    
    const REAL cutoff2 = cutoff*cutoff;

#pragma omp parallel for default(none) schedule(dynamic)\
shared(NLIST, NNEIG, P, qlist, CRL, tmp_offset) \
reduction(+ : totaln) reduction(min: err)
    for(INT64 px=0 ; px<NPART ; px++){
        const INT64 my_cell = CRL[px];
        // NNEIG[px] contains the number of neighbours of particle px. The start position is px*stride, neighbours
        // are consecutive.
        INT64 nn = 0;
        const INT64 ndx = STRIDE*px;
        // loop over cells.

        for( INT64 k=0 ; k<27 ; k++){
            INT64 jx = qlist[qoffset + my_cell + tmp_offset[k]]; //get first particle in other cell.
            while(jx > -1){
                if (px != jx) {
                    const REAL rj0 = P[jx*3]   - P[px*3    ];
                    const REAL rj1 = P[jx*3+1] - P[px*3 + 1];
                    const REAL rj2 = P[jx*3+2] - P[px*3 + 2];

                    // check if close enough
                    if ( (rj0*rj0 + rj1*rj1+ rj2*rj2) <= cutoff2 ) {
                        NLIST[ndx+nn] = jx;
                        nn++;
                    }
                }
                jx = qlist[jx];
            }
        }

        if (nn > STRIDE) {printf("bad neighbour count\n"); err=-1;}
        if (ndx+nn > STRIDE*NPART) {printf("bad neighbour index\n"); err=-2;}

        NNEIG[px] = nn;
        totaln += nn;
    }

    *total_num_neigh = totaln;

    return err;
}


