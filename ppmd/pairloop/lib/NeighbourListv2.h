





    #define PP (1)

    //cout << "------------------------------" << endl;
    //printf("start P[0] = %%f \\n", P[0]);


    const double cutoff = CUTOFF[0];
    const long max_len = MAX_LEN[0];

    const int _h_map[14][3] = {
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

    int tmp_offset[14];

    for(int ix=0; ix<14; ix++){
        tmp_offset[ix] = _h_map[ix][0] +
                         _h_map[ix][1] * CA[0] +
                         _h_map[ix][2] * CA[0]* CA[1];
    }

    const int _s_h_map[13][3] = {
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

    int selective_lookup[13];
    int s_tmp_offset[13];

    for( int ix = 0; ix < 13; ix++){
        selective_lookup[ix] = pow(2, ix);

        s_tmp_offset[ix] = _s_h_map[ix][0] +
                           _s_h_map[ix][1] * CA[0] +
                           _s_h_map[ix][2] * CA[0]* CA[1];
    }


    const double _b0 = B[0];
    const double _b2 = B[2];
    const double _b4 = B[4];
    
    //cout << "boundary" << endl;
    //cout << B[0] << " " << B[1] << endl;
    //cout << B[2] << " " << B[3] << endl;
    //cout << B[4] << " " << B[5] << endl;


    const double _icel0 = 1.0/CEL[0];
    const double _icel1 = 1.0/CEL[1];
    const double _icel2 = 1.0/CEL[2];

    const int _ca0 = CA[0];
    const int _ca1 = CA[1];
    const int _ca2 = CA[2];


    // loop over particles
    long m = -1;
    for (int ix=0; ix<end_ix; ix++) {

        const double pi0 = P[ix*3];
        const double pi1 = P[ix*3 + 1];
        const double pi2 = P[ix*3 + 2];

        const int val = CRL[ix];

        const int C0 = val %% _ca0;
        const int C1 = ((val - C0) / _ca0) %% _ca1;
        const int C2 = (((val - C0) / _ca0) - C1 ) / _ca1;
        if (val != ((C2*_ca1 + C1)*_ca0 + C0) ) {cout << "CELL FAILURE, val=" << val << " 0 " << C0 << " 1 " << C1 << " 2 " << C2 << endl;}


        //cout << "val = " << val << " C0 = " << C0 << " C1 = " << C1 << " C2 = " << C2 << endl;
        //cout << " Ca0 = " << _ca0 << " Ca1 = " << _ca1 << " Ca2 = " << _ca2 << endl;


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

        //cout << "flag " << flag << endl;

        if (flag > 0) {

            //check the possble 13 directions
            for( int csx = 0; csx < 13; csx++){
                if (flag & selective_lookup[csx]){
                    
                    //cout << "S look " << csx << endl;

                    int iy = q[n + val + s_tmp_offset[csx]];
                    while(iy > -1){

                        const double rj0 = P[iy*3]   - pi0;
                        const double rj1 = P[iy*3+1] - pi1;
                        const double rj2 = P[iy*3+2] - pi2;

                        //cout << "S_iy = " << iy << " py0 = " << P[iy*3+0] << " py1 = " << P[iy*3+1] << " py2 = " << P[iy*3+2] << endl;


                        if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {

                            m++;
                            if (m >= max_len){
                                RC[0] = -1;
                                return;
                            }

                            NEIGHBOUR_LIST[m] = iy;
                        }

                    iy=q[iy]; }
                }
            }

            //printf(" ##\\n");

        }

        // standard directions

        for(int k = 0; k < 14; k++){
            
            //cout << "\\toffset: " << k << endl;

            int iy = q[n + val + tmp_offset[k]];
            while (iy > -1) {

                if ( (tmp_offset[k] != 0) || (iy > ix) ){

                    //if (k==12){ cout << "iy=" << iy << endl;}

                    const double rj0 = P[iy*3]   - pi0;
                    const double rj1 = P[iy*3+1] - pi1;
                    const double rj2 = P[iy*3+2] - pi2;
                    
                    //if (k==12){ cout << "iy=" << iy << " y= " << P[iy*3+0] << " y= " << P[iy*3+1] << " y=" << P[iy*3+2] << endl;}


                    if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {

                        m++;
                        if (m >= max_len){
                            RC[0] = -1;
                            return;
                        }

                        NEIGHBOUR_LIST[m] = iy;
                    }

                }

            iy = q[iy]; }

        }
    }
    NEIGHBOUR_STARTS[end_ix] = m + 1;

    RC[0] = 0;


    //printf("end P[0] = %%f \\n", P[0]);

    //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    return;
    ''' % {'NULL': ''}



    _dat_dict = {'B': self._domain.boundary,              # Inner boundary on local domain (inc halo cells)
                 'P': self._positions,                    # positions
                 'CEL': self._domain.cell_edge_lengths,   # cell edge lengths
                 'CA': self._domain.cell_array,           # local domain cell array
                 'q': self.cell_list.cell_list,           # cell list
                 'CRL': self.cell_list.cell_reverse_lookup,
                 'CUTOFF': self.cell_width_squared,
                 'NEIGHBOUR_STARTS': self.neighbour_starting_points,
                 'NEIGHBOUR_LIST': self.list,
                 'MAX_LEN': self.max_len,
                 'RC': self._return_code}

    _static_args = {'end_ix': ct.c_int,  # Number of particles.
                    'n': ct.c_int}       # start of cell point in list.
