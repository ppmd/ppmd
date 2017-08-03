





extern "C"
int NeighbourListNonN3(
    const INT end_ix,  // Number of particles.
    const INT n,       // start of cell point in list.
    const REAL * RESTRICT P,    // positions
    const REAL * RESTRICT CA,   // local domain cell array
    const INT * RESTRICT q,     //  cell list
    const INT * RESTRICT CRL,   // cell reverse lookup
    const REAL * RESTRICT CUTOFF,     // cutoff squared
    LONG * RESTRICT NEIGHBOUR_STARTS, // self.neighbour_starting_points,
    INT * RESTRICT NEIGHBOUR_LIST,    // self.list,
    const LONG * RESTRICT MAX_LEN,     // self.max_len,
    INT * RESTRICT RC                 // self._return_code}
){

    const REAL cutoff = CUTOFF[0];
    const LONG max_len = MAX_LEN[0];

    const int _h_map[27][3] = {
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

    int tmp_offset[27];
    for(int ix=0; ix<27; ix++){
        tmp_offset[ix] = _h_map[ix][0] + _h_map[ix][1] * CA[0] + _h_map[ix][2] * CA[0]* CA[1];
    }

    // loop over particles
    LONG m = -1;

    for (INT ix=0; ix<end_ix; ix++) {

        const INT val = CRL[ix];

        //NEIGHBOUR_STARTS[ix] = m + (LONG)1;

        printf("%%d\n", m);
        printf("%%f %%f %%f\n", P[ix*3], P[ix*3+1], P[ix*3+2]);

        for(int k = 0; k < 27; k++){

            INT iy = q[n + val + tmp_offset[k]];
            while (iy > -1) {


                        printf("%%d, %%d\n", ix, iy);
                if (ix != iy){

                    const REAL rj0 = P[iy*3] - P[ix*3];
                    const REAL rj1 = P[iy*3+1] - P[ix*3 + 1];
                    const REAL rj2 = P[iy*3+2] - P[ix*3 + 2];

                    if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {

                        //printf("ix=%%f %%f %%f\n", P[ix*3], P[ix*3+1], P[ix*3+2]);
                        //printf("%%d-%%f %%f %%f\n",iy, P[iy*3], P[iy*3+1], P[iy*3+2]);
                        //printf("R=%%f\n", (rj0*rj0 + rj1*rj1 + rj2*rj2));


                        m++;
                        if (m < max_len){
                            //NEIGHBOUR_LIST[m] = iy;
                        } else {
                            RC[0] = -1;
                            return -1;
                        }
                    }

                }

                iy = q[iy];
            }

        }
    }
    NEIGHBOUR_STARTS[end_ix] = m + (LONG)1;
    RC[0] = 0;
    return 0;
}





















