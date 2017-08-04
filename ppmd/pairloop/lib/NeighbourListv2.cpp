

extern "C"
int NeighbourListv2(
    const INT end_ix,   // Number of particles.
    const INT n,        // start of cell point in list.
    const REAL * RESTRICT B,   // Inner boundary on local domain (inc halo cells)
    const REAL * RESTRICT P,
    const REAL * RESTRICT CEL,
    const INT * RESTRICT CA,
    const INT * RESTRICT q,
    const INT * RESTRICT CRL,
    const REAL * RESTRICT CUTOFF,
    LONG * RESTRICT NEIGHBOUR_STARTS,
    INT * RESTRICT NEIGHBOUR_LIST,
    LONG * RESTRICT MAX_LEN,
    INT * RESTRICT RC
){

    const REAL cutoff = CUTOFF[0];
    const LONG max_len = MAX_LEN[0];

    const INT _h_map[14][3] = {
                                        { 0 , 0 , 0 },
                                        { 0 , 1 , 0 },
                                        { 1 , 0 , 0 },
                                        { 1 , 1 , 0 },
                                        { 1 ,-1 , 0 },

                                        {-1 , 0 , 1 },
                                        {-1 , 1 , 1 },
                                        {-1 ,-1 , 1 },
                                        { 0 , 0 , 1 },
                                        { 0 , 1 , 1 },
                                        { 0 ,-1 , 1 },
                                        { 1 , 0 , 1 },
                                        { 1 , 1 , 1 },
                                        { 1 ,-1 , 1 }
                                    };

    INT tmp_offset[14];

    for(int ix=0; ix<14; ix++){
        tmp_offset[ix] = _h_map[ix][0] +
                         _h_map[ix][1] * CA[0] +
                         _h_map[ix][2] * CA[0]* CA[1];
    }

    const INT _s_h_map[13][3] = {
                                        {-1 ,-1 ,-1 },
                                        { 0 ,-1 ,-1 },
                                        { 1 ,-1 ,-1 },
                                        {-1 , 0 ,-1 },
                                        { 0 , 0 ,-1 },
                                        { 1 , 0 ,-1 },
                                        {-1 , 1 ,-1 },
                                        { 0 , 1 ,-1 },
                                        { 1 , 1 ,-1 },

                                        {-1 ,-1 , 0 },
                                        { 0 ,-1 , 0 },
                                        {-1 , 0 , 0 },
                                        {-1 , 1 , 0 }
                                    };

    INT selective_lookup[13];
    INT s_tmp_offset[13];

    for( int ix = 0; ix < 13; ix++){
        selective_lookup[ix] = pow(2, ix);

        s_tmp_offset[ix] = _s_h_map[ix][0] +
                           _s_h_map[ix][1] * CA[0] +
                           _s_h_map[ix][2] * CA[0]* CA[1];
    }

    const REAL _b0 = B[0];
    const REAL _b2 = B[2];
    const REAL _b4 = B[4];

    const REAL _icel0 = 1.0/CEL[0];
    const REAL _icel1 = 1.0/CEL[1];
    const REAL _icel2 = 1.0/CEL[2];

    const INT _ca0 = CA[0];
    const INT _ca1 = CA[1];
    const INT _ca2 = CA[2];

    // loop over particles
    LONG m = -1;
    for (int ix=0; ix<end_ix; ix++) {

        const REAL pi0 = P[ix*3];
        const REAL pi1 = P[ix*3 + 1];
        const REAL pi2 = P[ix*3 + 2];

        const INT val = CRL[ix];

        const INT C0 = val %% _ca0;
        const INT C1 = ((val - C0) / _ca0) %% _ca1;
        const INT C2 = (((val - C0) / _ca0) - C1 ) / _ca1;
        if (val != ((C2*_ca1 + C1)*_ca0 + C0) ) {cout << "CELL FAILURE, val=" << val << " 0 " << C0 << " 1 " << C1 << " 2 " << C2 << endl;}

        NEIGHBOUR_STARTS[ix] = m + 1;

        // non standard directions
        // selective stencil lookup into halo

        int flag = 0;
        if ( C0 == 1 ) { flag |= 6729; }
        if ( C0 == (_ca0 - 2) ) { flag |= 292; }
        if ( C1 == 1 ) { flag |= 1543; }
        if ( C1 == (_ca1 - 2) ) { flag |= 4544; }
        if ( C2 == 1 ) { flag |= 511; }

        // if flag > 0 then we are near a halo
        // that needs attention
        if (flag > 0) {
            //check the possble 13 directions
            for( int csx = 0; csx < 13; csx++){
                if (flag & selective_lookup[csx]){
                    INT iy = q[n + val + s_tmp_offset[csx]];
                    while(iy > -1){
                        const REAL rj0 = P[iy*3]   - pi0;
                        const REAL rj1 = P[iy*3+1] - pi1;
                        const REAL rj2 = P[iy*3+2] - pi2;
                        if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {
                            m++;
                            if (m >= max_len){
                                RC[0] = -1;
                                return -1;
                            }
                            NEIGHBOUR_LIST[m] = iy;
                        }
                    iy=q[iy]; }
                }
            }
        }

        // standard directions
        for(int k = 0; k < 14; k++){
            INT iy = q[n + val + tmp_offset[k]];
            while (iy > -1) {
                if ( (tmp_offset[k] != 0) || (iy > ix) ){
                    const REAL rj0 = P[iy*3]   - pi0;
                    const REAL rj1 = P[iy*3+1] - pi1;
                    const REAL rj2 = P[iy*3+2] - pi2;
                    if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {
                        m++;
                        if (m >= max_len){
                            RC[0] = -1;
                            return -1;
                        }
                        NEIGHBOUR_LIST[m] = iy;
                    }
                }
            iy = q[iy]; }
        }
    }
    NEIGHBOUR_STARTS[end_ix] = m + 1;

    RC[0] = 0;
    return 0;
}




