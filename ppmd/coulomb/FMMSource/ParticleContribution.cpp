static inline void global_local_cell_tuple(
    const INT64 * RESTRICT cube_offset,
    const REAL * RESTRICT cube_inverse_len,
    const REAL * RESTRICT boundary,
    const REAL px,
    const REAL py,
    const REAL pz,
    INT64 * RESTRICT gcx,
    INT64 * RESTRICT gcy,
    INT64 * RESTRICT gcz,
    INT64 * RESTRICT cx,
    INT64 * RESTRICT cy,
    INT64 * RESTRICT cz
){
    // these are slowest to fastest
    INT64 cox = cube_offset[2];
    INT64 coy = cube_offset[1];
    INT64 coz = cube_offset[0];

    const REAL pxs = px + 0.5 * boundary[0];
    const REAL pys = py + 0.5 * boundary[1];
    const REAL pzs = pz + 0.5 * boundary[2];
    
    // compute global cell as a REAL value
    const REAL rgx = pxs * cube_inverse_len[0];
    const REAL rgy = pys * cube_inverse_len[1];
    const REAL rgz = pzs * cube_inverse_len[2];
    
    // compute local cell as a INT64 value, from a real value
    *cx = (INT64) (rgx - ((REAL)cube_offset[2]));
    *cy = (INT64) (rgy - ((REAL)cube_offset[1]));
    *cz = (INT64) (rgz - ((REAL)cube_offset[0]));
    
    // compute global cell from local value
    *gcx = *cx + cox;
    *gcy = *cy + coy;
    *gcz = *cz + coz;
    
    // this kills the vectorisation but should never be triggered
    // which avoids vectorisation effecting the cell binning
    // there is almost certainly a better solution that is portable
    if ( (abs(px - boundary[0] - 1) < 0.01) || (abs(py - boundary[1] - 1) < 0.01) || (abs(pz - boundary[2] - 1) < 0.01) ) {
        printf("px %f, pxs %f, rgx %f, cx %d, gcx %d\n", px, pxs, rgx, *cx, *gcx);
        printf("py %f, pys %f, rgy %f, cy %d, gcy %d\n", py, pys, rgy, *cy, *gcy);
        printf("pz %f, pzs %f, rgz %f, cz %d, gcz %d\n", pz, pzs, rgz, *cz, *gcz);
    }
}


static inline void global_cell_tuple(
    const REAL * RESTRICT cube_inverse_len,
    const REAL px,
    const REAL py,
    const REAL pz,
    const REAL * RESTRICT boundary,
    INT64 * RESTRICT cxt,
    INT64 * RESTRICT cyt,
    INT64 * RESTRICT czt
){  
    // shift position as origin is centered then
    // bin into cells
    // boundary here is the extent
    const REAL pxs = px + 0.5 * boundary[0];
    const REAL pys = py + 0.5 * boundary[1];
    const REAL pzs = pz + 0.5 * boundary[2];

    *cxt = pxs*cube_inverse_len[0];
    *cyt = pys*cube_inverse_len[1];
    *czt = pzs*cube_inverse_len[2];
}

static inline INT64 local_cell_tuple(
    const INT64 gcx, 
    const INT64 gcy, 
    const INT64 gcz, 
    const INT64 * RESTRICT cube_offset,
    INT64 * RESTRICT cx,
    INT64 * RESTRICT cy,
    INT64 * RESTRICT cz
){
    // local index is given by global index minus offset
    *cx = gcx - ((INT64)cube_offset[2]);
    *cy = gcy - ((INT64)cube_offset[1]);
    *cz = gcz - ((INT64)cube_offset[0]);
}

static inline INT64 drift_compensation(
    const INT64 * RESTRICT cube_dim,
    INT64 * RESTRICT gcx,
    INT64 * RESTRICT gcy,
    INT64 * RESTRICT gcz,
    INT64 * RESTRICT cx,
    INT64 * RESTRICT cy,
    INT64 * RESTRICT cz
){
    if (((*cx) > ((INT64)cube_dim[2])) || ((*cy) > ((INT64)cube_dim[1])) || ((*cz) > ((INT64)cube_dim[0])) ){
        printf("Error: Particle far from subdomain. %ld>%ld | %ld>%ld | %ld>%ld \n", 
                *cx , cube_dim[2] , *cy , cube_dim[1] , *cz , cube_dim[0]);
        return (INT64) -1;}
    else if (((*cx) < 0) || ((*cy) < 0) || ((*cz) <0) ) {
        printf("Error: Particle far from subdomain. %ld<%ld | %ld<%ld | %ld<%ld \n", 
                *cx , 0 , *cy , 0 , *cz , 0);
        return (INT64) -1;
    }
    
    // Reusing neighbourlists allows particles to drift small distances out of a
    // subdomain.
    if (*cx == ((INT64)cube_dim[2])){ *cx -= 1; *gcx -= 1; }
    if (*cy == ((INT64)cube_dim[1])){ *cy -= 1; *gcy -= 1; }
    if (*cz == ((INT64)cube_dim[0])){ *cz -= 1; *gcz -= 1; }
    // as casting to int truncates towards zero we can pass on checking the lower 
    // bound
    return 0;
}


static inline INT64 compute_cell(
    const REAL * RESTRICT cube_inverse_len,
    const REAL px,
    const REAL py,
    const REAL pz,
    const REAL * RESTRICT boundary,
    const INT64 * RESTRICT cube_offset,
    const INT64 * RESTRICT cube_dim,
    const INT64 * RESTRICT cube_side_counts,   
    INT64 * global_cell
){
    // global cell tuple
    INT64 gcx, gcy, gcz, cx, cy, cz;
    global_local_cell_tuple(cube_offset, cube_inverse_len, boundary,
            px, py, pz, &gcx, &gcy, &gcz, &cx, &cy, &cz);

    //global_cell_tuple(cube_inverse_len, px, py, pz, boundary, &gcx, &gcy, &gcz);
    //local_cell_tuple(gcx, gcy, gcz, cube_offset, &cx, &cy, &cz);

    if (drift_compensation(cube_dim, &gcx, &gcy, &gcz, &cx, &cy, &cz) < 0){ 
        printf("position: %.16f %.16f %.16f\nboundary %.16f %.16f %.16f\ninv cell %.16f %.16f %.16f\noffset %ld %ld %ld\n", 
                px, py, pz,
                boundary[0], boundary[1], boundary[2], 
                cube_inverse_len[0], cube_inverse_len[1], cube_inverse_len[2],
                cube_offset[0], cube_offset[1], cube_offset[2]);
        return -1; 
    }

    *global_cell = ((INT64) (  gcx + cube_side_counts[2] * (gcy + cube_side_counts[1] * gcz )  ));
    if ((*global_cell) < 0){
        printf("err: Negative global cell\n");
        return -1;
    }

    return cx + cube_dim[2] * (cy + cube_dim[1] * cz);
}

static inline INT64 compute_cell_spherical(
    const REAL * RESTRICT cube_inverse_len,
    const REAL * RESTRICT cube_half_len,
    const REAL px,
    const REAL py,
    const REAL pz,
    const REAL * RESTRICT boundary,
    const INT64 * RESTRICT cube_offset,
    const INT64 * RESTRICT cube_dim,
    REAL * RESTRICT radius,
    REAL * RESTRICT ctheta,
    REAL * RESTRICT cphi,
    REAL * RESTRICT sphi,
    REAL * RESTRICT msphi
){

    INT64 gcx, gcy, gcz, cx, cy, cz;

    global_local_cell_tuple(cube_offset, cube_inverse_len, boundary,
            px, py, pz, &gcx, &gcy, &gcz, &cx, &cy, &cz);

    // global_cell_tuple(cube_inverse_len, px, py, pz, boundary, &gcx, &gcy, &gcz);
    // local_cell_tuple(gcx, gcy, gcz, cube_offset, &cx, &cy, &cz);

    if (drift_compensation(cube_dim, &gcx, &gcy, &gcz, &cx, &cy, &cz) < 0){
        printf("position: %f %f %f\nboundary %f %f %f\n", 
                px, py, pz, boundary[0], boundary[1], boundary[2]);

        return -1; 
    }

    const int cell_idx = cx + cube_dim[2] * (cy + cube_dim[1] * cz);
    // now compute the vector between particle and cell center in spherical coords
    // cell center
    const REAL ccx = (gcx * 2 + 1) * cube_half_len[0] - 0.5 * boundary[0]; 
    const REAL ccy = (gcy * 2 + 1) * cube_half_len[1] - 0.5 * boundary[1]; 
    const REAL ccz = (gcz * 2 + 1) * cube_half_len[2] - 0.5 * boundary[2]; 

    // compute Cartesian displacement vector
    const REAL dx = px - ccx;
    const REAL dy = py - ccy;
    const REAL dz = pz - ccz;

    // convert to spherical
    const REAL dx2 = dx*dx;
    const REAL dx2_p_dy2 = dx2 + dy*dy;
    const REAL d2 = dx2_p_dy2 + dz*dz;
    *radius = sqrt(d2);

    //printf("\t%f\t%f\t%f global cell %d\t%d\t%d | radius %f |\t%f\t%f\t%f\n", px, py, pz, gcx, gcy, gcz, *radius, ccx, ccy, ccz);
    
    const REAL theta = atan2(sqrt(dx2_p_dy2), dz);
    *ctheta = cos(theta);
    
    const REAL phi = atan2(dy, dx);

    *cphi = cos(phi);
    *sphi = sin(phi); 
    *msphi = -1.0 * (*sphi);
    return cell_idx;
}

static inline void next_pos_exp(
    const REAL cphi,
    const REAL sphi,
    const REAL rpe,
    const REAL ipe,
    REAL * RESTRICT nrpe,
    REAL * RESTRICT nipe
){
    *nrpe = rpe*cphi - ipe*sphi;
    *nipe = rpe*sphi + ipe*cphi;
}

static inline void next_neg_exp(
    const REAL cphi,
    const REAL msphi,
    const REAL rne,
    const REAL ine,
    REAL * RESTRICT nrne,
    REAL * RESTRICT nine
){
    *nrne = rne*cphi  - ine*msphi;
    *nine = rne*msphi + ine*cphi;
}

static inline REAL m1_m(int m){
    return 1.0 - 2.0*((REAL)(m & 1));
}


extern "C"
INT64 particle_contribution(
    const INT64 nlevel,
    const INT64 npart,
    const INT64 thread_max,
    const REAL * RESTRICT position,             // xyz
    const REAL * RESTRICT charge,
    INT64 * RESTRICT fmm_cell,
    const REAL * RESTRICT boundary,             // xl. xu, yl, yu, zl, zu
    const INT64 * RESTRICT cube_offset,        // zyx (slowest to fastest)
    const INT64 * RESTRICT cube_dim,           // as above
    const INT64 * RESTRICT cube_side_counts,   // as above
    REAL * RESTRICT cube_data,                  // lexicographic
    INT64 * RESTRICT thread_assign
){
    INT64 err = 0;
    omp_set_num_threads(thread_max);
    //printf("%f | %f | %f\n", boundary[0], boundary[1], boundary[2]);


    const REAL cube_ilen[3] = {
        cube_side_counts[2]/boundary[0],
        cube_side_counts[1]/boundary[1],
        cube_side_counts[0]/boundary[2]
    };
    
    const REAL cube_half_side_len[3] = {
        0.5*boundary[0]/cube_side_counts[2],
        0.5*boundary[1]/cube_side_counts[1],
        0.5*boundary[2]/cube_side_counts[0]
    };

    // loop over particle and assign them to threads based on cell
    #pragma omp parallel for default(none) shared(thread_assign, position, \
    boundary, cube_offset, cube_dim, cube_ilen, charge, fmm_cell, cube_side_counts) \
    reduction(min: err)
    for(INT64 ix=0 ; ix<npart ; ix++){

        INT64 global_cell = -1;
        const INT64 ix_cell = compute_cell(cube_ilen, position[ix*3], position[ix*3+1], 
                position[ix*3+2], boundary, cube_offset, cube_dim, cube_side_counts, &global_cell);

        if (ix_cell < 0) {
            err = -1;
        } else {
            const INT64 thread_owner = ix_cell % thread_max;
            INT64 thread_layer;
            #pragma omp atomic capture
            thread_layer = ++thread_assign[thread_owner];

            thread_assign[thread_max + npart*thread_owner + thread_layer - 1] = ix;

            // assign this particle a cell for short range part
            fmm_cell[ix] = global_cell;
            //printf("ix = %d global_cell = %d\n", ix, global_cell);
        }
    }
    if (err < 0) { return err; }
 
    // check all particles were assigned to a thread
    INT64 check_sum = 0;
    for(INT64 ix=0 ; ix<thread_max ; ix++){
        check_sum += thread_assign[ix]; 
        //printf("tx %d val %d\n", ix, thread_assign[ix]);
    }
    if (check_sum != npart) {printf("npart %d assigned %d\n", npart, check_sum); return -2;}
    
    // Above we assigned particles to threads in a way that avoids needing atomics here.
    // Computing the cell is duplicate work, but the computation is cheap and the positions
    // needed to be read anyway to compute the moments.

    // using omp parallel for with schedule(static, 1) is robust against the omp
    // implementation deciding to launch less threads here than thread_max (which it
    // is entitled to do).
    
    REAL exp_space[thread_max][nlevel*4 + 2];
    // pre compute factorial and double factorial
    const INT64 nfact = (2*nlevel > 4) ? 2*nlevel : 4;
    REAL factorial_vec[nfact];
    REAL double_factorial_vec[nfact];

    factorial_vec[0] = 1.0;
    factorial_vec[1] = 1.0;
    factorial_vec[2] = 2.0;
    double_factorial_vec[0] = 1.0;
    double_factorial_vec[1] = 1.0;
    double_factorial_vec[2] = 2.0;

    for( INT64 lx=3 ; lx<nfact ; lx++ ){
        factorial_vec[lx] = lx * factorial_vec[lx-1];
        double_factorial_vec[lx] = lx * double_factorial_vec[lx-2];
    }
    
    REAL P_SPACE_VEC[thread_max][nlevel*nlevel*2];


    INT64 count = 0;
    #pragma omp parallel for default(none) shared(thread_assign, position, boundary, \
        cube_offset, cube_dim, err, cube_data, exp_space, factorial_vec, double_factorial_vec, P_SPACE_VEC, \
        cube_half_side_len, cube_ilen, charge) \
        schedule(static,1) \
        reduction(+: count)
    for(INT64 tx=0 ; tx<thread_max ; tx++){
        const int tid = omp_get_thread_num();
        const INT64 ncomp = nlevel*nlevel*2;
        REAL * P_SPACE = P_SPACE_VEC[tid];
        for(INT64 px=0 ; px<thread_assign[tx] ; px++){
            INT64 ix = thread_assign[thread_max + npart*tx + px];
            REAL radius, ctheta, cphi, sphi, msphi;
            const INT64 ix_cell = compute_cell_spherical(
                cube_ilen, cube_half_side_len, position[ix*3], position[ix*3+1], 
                position[ix*3+2], boundary, cube_offset, cube_dim,
                &radius, &ctheta, &cphi, &sphi, &msphi
            );
            
            //if (ix_cell == 464){
            //    printf("%d \t radius: %f cos(theta): %f cos(phi): %f sin(phi): %f -1*sin(phi): %f\n", 
            //    ix, radius, ctheta, cphi, sphi, msphi);
            //}
            
            if (tx != ix_cell % thread_max) {           
                #pragma omp critical
                {err = -3;}
            }

            //compute spherical harmonic moments

            // start with the complex exponential (will not vectorise)
            REAL * RESTRICT exp_vec = exp_space[tid]; 
            REAL * RESTRICT cube_start = &cube_data[ix_cell*ncomp];
            REAL * RESTRICT cube_start_im = &cube_data[ix_cell*ncomp + nlevel*nlevel];


            exp_vec[EXP_RE_IND(nlevel, 0)] = 1.0;
            exp_vec[EXP_IM_IND(nlevel, 0)] = 0.0;
            for (INT64 lx=1 ; lx<=((INT64)nlevel) ; lx++){
                next_pos_exp(
                    cphi, sphi,
                    exp_vec[EXP_RE_IND(nlevel, lx-1)],
                    exp_vec[EXP_IM_IND(nlevel, lx-1)],
                    &exp_vec[EXP_RE_IND(nlevel, lx)],
                    &exp_vec[EXP_IM_IND(nlevel, lx)]);
                next_neg_exp(
                    cphi, msphi,
                    exp_vec[EXP_RE_IND(nlevel, -1*(lx-1))],
                    exp_vec[EXP_IM_IND(nlevel, -1*(lx-1))],
                    &exp_vec[EXP_RE_IND(nlevel, -1*lx)],
                    &exp_vec[EXP_IM_IND(nlevel, -1*lx)]);
            }

            const REAL sqrt_1m2lx = sqrt(1.0 - ctheta*ctheta);

            // P_0^0 = 1;
            P_SPACE[P_SPACE_IND(nlevel, 0, 0)] = 1.0;
            
            for( int lx=0 ; lx<((int)nlevel) ; lx++){

                //compute the (lx+1)th P values using the lx-th values
                if (lx<(nlevel-1) ){ 
                    P_SPACE[P_SPACE_IND(nlevel, lx+1, lx+1)] = (-1.0 - 2.0*lx) * sqrt_1m2lx * \ 
                        P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

                    P_SPACE[P_SPACE_IND(nlevel, lx+1, lx)] = ctheta * (2*lx + 1) * \
                        P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

                    for( int mx=0 ; mx<lx ; mx++ ){
                        P_SPACE[P_SPACE_IND(nlevel, lx+1, mx)] = \
                            (ctheta * (2.0*lx+1.0) * P_SPACE[P_SPACE_IND(nlevel, lx, mx)] - \
                            (lx+mx)*P_SPACE[P_SPACE_IND(nlevel, lx-1, mx)]) / (lx - mx + 1);
                    }

                }
            }
            
            REAL rhol = 1.0;
            //loop over l and m
            for( int lx=0 ; lx<((int)nlevel) ; lx++ ){
                rhol = (lx > 0) ? rhol*radius : 1.0;

                for( int mx=-1*lx ; mx<=lx ; mx++ ){
                    const INT64 abs_mx = ABS(mx);
                    const REAL coeff = sqrt(factorial_vec[lx - abs_mx]/factorial_vec[lx + abs_mx]) \
                                       * charge[ix] * rhol;

                    const REAL plm = P_SPACE[P_SPACE_IND(nlevel, lx, abs_mx)];
                    
                    // add this particle's contribution to the cell expansion
                    cube_start[CUBE_IND(lx, mx)] += coeff * plm * exp_vec[EXP_RE_IND(nlevel, -1*mx)];
                    cube_start_im[CUBE_IND(lx, mx)] += coeff * plm * exp_vec[EXP_IM_IND(nlevel, -1*mx)];

                }
            }

            count++;
        }
    }   
    
    if (count != npart) {err = -4;}
    return err;
}









