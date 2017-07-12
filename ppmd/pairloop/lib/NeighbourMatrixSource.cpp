

extern "C"
signed long long OMPNeighbourMatrix(
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
){
    signed long long totaln = 0;
    signed long long err = 0;
    int tmp_offset[27];
    for(int ix=0; ix<27; ix++){
        tmp_offset[ix] = h_map[ix][0] + h_map[ix][1] * CA[0] + h_map[ix][2] * CA[0]* CA[1];
    }

#pragma omp parallel for default(none) schedule(static) shared(NLIST, NNEIG, P, qlist, CRL, tmp_offset) reduction(+ : totaln)
    for(int px=0 ; px<NPART ; px++){
        const int my_cell = CRL[px];
        // NNEIG[px] contains the number of neighbours of particle px. The start position is px*stride, neighbours
        // are consecutive.
        unsigned long long nn = 0;
        const unsigned long long ndx = ((unsigned long long)STRIDE)*((unsigned long long)px);
        // loop over cells.

        for( int k=0 ; k<27 ; k++){
            int jx = qlist[qoffset + my_cell + tmp_offset[k]]; //get first particle in other cell.
            while(jx > -1){
                if (px != jx) {
                    const double rj0 = P[jx*3]   - P[px*3    ];
                    const double rj1 = P[jx*3+1] - P[px*3 + 1];
                    const double rj2 = P[jx*3+2] - P[px*3 + 2];

                    // check if close enough
                    if ( (rj0*rj0 + rj1*rj1+ rj2*rj2) <= cutoff ) {
                        NLIST[ndx+((unsigned long long)nn)] = jx;
                        nn++;
                    }
                }
                jx = qlist[jx];
            }
        }

        if (nn > STRIDE) {printf("bad neighbour count\n");}
        if (ndx+((unsigned long long)nn) > STRIDE*NPART) {printf("bad neighbour index\n");}

        NNEIG[px] = (int) nn;
        totaln += nn;
    }

    err = totaln;
    return err;
}


