

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
){
    int totaln = 0;
    int tmp_offset[27];
    for(int ix=0; ix<27; ix++){
        tmp_offset[ix] = h_map[ix][0] + h_map[ix][1] * CA[0] + h_map[ix][2] * CA[0]* CA[1];
    }

#pragma omp parallel for default(none) shared(NLIST, NNEIG, P, qlist, CRL, tmp_offset) reduction(+ : totaln)
    for(int px=0 ; px<NPART ; px++){
        const int my_cell = CRL[px];
        // NNEIG[px] contains the number of neighbours of particle px. The start position is px*stride, neighbours
        // are consecutive.
        int nn = 0;
        const int ndx = STRIDE*px;
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
                        NLIST[ndx+nn] = jx;
                        nn++;
                    }
                }
                jx = qlist[jx];
            }
        }
        NNEIG[px] = nn;
        totaln += nn;
    }

    return totaln;
}


