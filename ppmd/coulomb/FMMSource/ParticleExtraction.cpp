static inline void cplx_mul_add(
    const REAL a,
    const REAL b,
    const REAL x,
    const REAL y,
    REAL * RESTRICT g,
    REAL * RESTRICT h
){
   // ( a + bi) * (x + yi) = (ax - by) + (xb + ay)i
    *g += a * x - b * y;
    *h += x * b + a * y;
}

static inline INT64 compute_cell_spherical(
    const REAL * RESTRICT cube_inverse_len,
    const REAL * RESTRICT cube_half_len,
    const REAL px,
    const REAL py,
    const REAL pz,
    const REAL * RESTRICT boundary,
    const UINT64 * RESTRICT cube_offset,
    const UINT64 * RESTRICT cube_dim,
    REAL * RESTRICT radius,
    REAL * RESTRICT ctheta,
    REAL * RESTRICT cphi,
    REAL * RESTRICT sphi,
    REAL * RESTRICT msphi
){
    const REAL pxs = px + 0.5 * boundary[0];
    const REAL pys = py + 0.5 * boundary[1];
    const REAL pzs = pz + 0.5 * boundary[2];

    const UINT64 cxt = pxs*cube_inverse_len[0];
    const UINT64 cyt = pys*cube_inverse_len[1];
    const UINT64 czt = pzs*cube_inverse_len[2];

    const UINT64 cx = cxt - cube_offset[2];
    const UINT64 cy = cyt - cube_offset[1];
    const UINT64 cz = czt - cube_offset[0];

    if (cx >= cube_dim[2] || cy >= cube_dim[1] || cz >= cube_dim[0] ){
        return (INT64) -1;}
    const int cell_idx = cx + cube_dim[2] * (cy + cube_dim[1] * cz);
    // now compute the vector between particle and cell center in spherical coords
    // cell center
    const REAL ccx = (cxt * 2 + 1) * cube_half_len[0] - 0.5 * boundary[0]; 
    const REAL ccy = (cyt * 2 + 1) * cube_half_len[1] - 0.5 * boundary[1]; 
    const REAL ccz = (czt * 2 + 1) * cube_half_len[2] - 0.5 * boundary[2]; 
    //printf("cube_inverse_lens %f %f %f\n", cube_inverse_len[0], cube_inverse_len[1], cube_inverse_len[2]);

    //printf("cube_half_len %f %f %f %f\n", cube_half_len[0], cube_half_len[1], cube_half_len[2], (cyt * 2 + 1) + cube_half_len[1]);
    // compute Cartesian displacement vector
    
    const REAL dx = px - ccx;
    const REAL dy = py - ccy;
    const REAL dz = pz - ccz;
    
    // reverse INCORRECT, for testing
    //const REAL dx = ccx - px;
    //const REAL dy = ccy - py;
    //const REAL dz = ccz - pz;


    //printf("dx %f px %f mid %f\n", dx, px, ccx);
    //printf("dy %f py %f mid %f\n", dy, py, ccy);
    //printf("dz %f pz %f mid %f\n", dz, pz, ccz);
    

    // convert to spherical
    const REAL dx2 = dx*dx;
    const REAL dx2_p_dy2 = dx2 + dy*dy;
    const REAL d2 = dx2_p_dy2 + dz*dz;
    *radius = sqrt(d2);

    const REAL dx2_p_dy2_o_d2 = dx2_p_dy2 / d2;
    // theta part
    const REAL ct1 = isnormal(dx2_p_dy2_o_d2) ? sqrt(1.0 - dx2_p_dy2_o_d2) : 1.0;
    *ctheta = (dz >= 0.0) ? ct1 : -1.0 * ct1;



    // phi part
    const REAL sqrt_dx2pdy2 = sqrt(dx2_p_dy2);
    const REAL dx_o_sqrt_dx2pdy2 = dx / sqrt_dx2pdy2;
    *cphi = isnormal(dx_o_sqrt_dx2pdy2) ? dx_o_sqrt_dx2pdy2 : 1.0;
    const REAL dx2_o_dx2_p_dy2 = dx2/dx2_p_dy2;
    const REAL sp1 = isnormal(dx2_o_dx2_p_dy2) ? sqrt(1.0 - dx2_o_dx2_p_dy2) : 0.0;

    *sphi = (dy > 0) ? sp1 : -1.0 * sp1; 
    *msphi = -1.0 * (*sphi);
    return cell_idx;
}


static inline INT64 extract_cell_spherical(
    const REAL * RESTRICT cube_inverse_len,
    const REAL * RESTRICT cube_half_len,
    const REAL px,
    const REAL py,
    const REAL pz,
    const REAL * RESTRICT boundary,
    const UINT64 * RESTRICT cube_offset,
    const UINT64 * RESTRICT cube_dim,
    const UINT64 * RESTRICT cube_side_counts,   
    const INT32 global_cell,
    REAL * RESTRICT radius,
    REAL * RESTRICT ctheta,
    REAL * RESTRICT cphi,
    REAL * RESTRICT sphi,
    REAL * RESTRICT msphi
){
    
    const INT64 nsx = cube_side_counts[2];
    const INT64 nsy = cube_side_counts[1];
    const INT64 nsz = cube_side_counts[0];

    const INT64 cxt = global_cell % nsx;
    const INT64 cyt = ((global_cell - cxt) / nsx) % nsy;
    const INT64 czt = (global_cell - cxt - cyt*nsx) / (nsx*nsy);

    const UINT64 cx = cxt - cube_offset[2];
    const UINT64 cy = cyt - cube_offset[1];
    const UINT64 cz = czt - cube_offset[0];

    if (cx >= cube_dim[2] || cy >= cube_dim[1] || cz >= cube_dim[0] ){
        return (INT64) -1;}
    const int cell_idx = cx + cube_dim[2] * (cy + cube_dim[1] * cz);
    // now compute the vector between particle and cell center in spherical coords
    // cell center
    const REAL ccx = (cxt * 2 + 1) * cube_half_len[0] - 0.5 * boundary[0]; 
    const REAL ccy = (cyt * 2 + 1) * cube_half_len[1] - 0.5 * boundary[1]; 
    const REAL ccz = (czt * 2 + 1) * cube_half_len[2] - 0.5 * boundary[2]; 
    //printf("cube_inverse_lens %f %f %f\n", cube_inverse_len[0], cube_inverse_len[1], cube_inverse_len[2]);

    //printf("cube_half_len %f %f %f %f\n", cube_half_len[0], cube_half_len[1], cube_half_len[2], (cyt * 2 + 1) + cube_half_len[1]);
    // compute Cartesian displacement vector
    const REAL dx = px - ccx;
    const REAL dy = py - ccy;
    const REAL dz = pz - ccz;
    
    //printf("dx %f px %f mid %f\n", dx, px, ccx);
    //printf("dy %f py %f mid %f\n", dy, py, ccy);
    //printf("dz %f pz %f mid %f\n", dz, pz, ccz);
    

    // convert to spherical
    const REAL dx2 = dx*dx;
    const REAL dx2_p_dy2 = dx2 + dy*dy;
    const REAL d2 = dx2_p_dy2 + dz*dz;
    *radius = sqrt(d2);
    
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
INT32 particle_extraction(
    const INT64 nlevel,
    const UINT64 npart,
    const INT32 thread_max,
    const REAL * RESTRICT position,             // xyz
    const REAL * RESTRICT charge,
    const INT32 * RESTRICT fmm_cell,
    const REAL * RESTRICT boundary,             // xl. xu, yl, yu, zl, zu
    const UINT64 * RESTRICT cube_offset,        // zyx (slowest to fastest)
    const UINT64 * RESTRICT cube_dim,           // as above
    const UINT64 * RESTRICT cube_side_counts,   // as above
    const REAL * RESTRICT local_moments,
    REAL * RESTRICT phi_data,                  // lexicographic
    const INT32 * RESTRICT thread_assign
){
    INT32 err = 0;
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


    // Above we assigned particles to threads in a way that avoids needing atomics here.
    // Computing the cell is duplicate work, but the computation is cheap and the positions
    // needed to be read anyway to compute the moments.

    // using omp parallel for with schedule(static, 1) is robust against the omp
    // implementation deciding to launch less threads here than thread_max (which it
    // is entitled to do).
    
    REAL exp_space[thread_max][nlevel*4 + 2];
    // pre compute factorial and double factorial
    const UINT64 nfact = (2*nlevel > 4) ? 2*nlevel : 4;
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

    UINT32 count = 0;
    REAL potential_energy = 0.0;

    #pragma omp parallel for default(none) shared(thread_assign, position, \
        boundary, cube_offset, cube_dim, err, exp_space, \
        factorial_vec, double_factorial_vec, P_SPACE_VEC, \
        cube_half_side_len, cube_ilen, charge, local_moments, fmm_cell, cube_side_counts) \
        schedule(dynamic) \
        reduction(+: count) reduction(+: potential_energy)
    for(INT32 ix=0 ; ix<npart ; ix++){
        const int tid = omp_get_thread_num();
        const UINT64 ncomp = nlevel*nlevel*2;
        REAL * P_SPACE = P_SPACE_VEC[tid];
        REAL local_pe = 0.0;
            
            if (ix<0) {
                #pragma omp critical
                {err = -5;}
            } else if (ix >= npart) {
                #pragma omp critical
                {err = -6;}
            }


            REAL radius, ctheta, cphi, sphi, msphi;
            const INT64 ix_cell = extract_cell_spherical(
                cube_ilen, cube_half_side_len, position[ix*3], position[ix*3+1], 
                position[ix*3+2], boundary, cube_offset, cube_dim,
                cube_side_counts, fmm_cell[ix],
                &radius, &ctheta, &cphi, &sphi, &msphi
            );
            //compute spherical harmonic moments
            // start with the complex exponential (will not vectorise)
            REAL * RESTRICT exp_vec = exp_space[tid]; 
            const REAL * RESTRICT cube_start = &local_moments[ix_cell*ncomp];
            const REAL * RESTRICT cube_start_im = \
                &local_moments[ix_cell*ncomp + nlevel*nlevel];


            exp_vec[EXP_RE_IND(nlevel, 0)] = 1.0;
            exp_vec[EXP_IM_IND(nlevel, 0)] = 0.0;
            for (INT32 lx=1 ; lx<=((INT32)nlevel) ; lx++){
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
                    P_SPACE[P_SPACE_IND(nlevel, lx+1, lx+1)] = \ 
                    (-1.0 - 2.0*lx) * sqrt_1m2lx * \ 
                    P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

                    P_SPACE[P_SPACE_IND(nlevel, lx+1, lx)] = \
                        ctheta * (2*lx + 1) * \
                        P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

                    for( int mx=0 ; mx<lx ; mx++ ){
                        P_SPACE[P_SPACE_IND(nlevel, lx+1, mx)] = \
                            (ctheta * (2.0*lx+1.0) * \
                            P_SPACE[P_SPACE_IND(nlevel, lx, mx)] - \
                            (lx+mx)*P_SPACE[P_SPACE_IND(nlevel, lx-1, mx)])/ \
                            (lx - mx + 1);
                    }

                }
            }

            
            REAL rhol = 1.0;
            //loop over l and m
            for( int lx=0 ; lx<((int)nlevel) ; lx++ ){
                rhol = (lx > 0) ? rhol*radius : 1.0;

                for( int mx=-1*lx ; mx<=lx ; mx++ ){
                    const UINT32 abs_mx = abs(mx);
                    const REAL coeff = sqrt(factorial_vec[lx - abs_mx]/
                        factorial_vec[lx + abs_mx]) * charge[ix] * rhol;

                    const REAL plm = P_SPACE[P_SPACE_IND(nlevel, lx, abs_mx)];
                    const REAL ylm_re = coeff * plm * \
                        exp_vec[EXP_RE_IND(nlevel, mx)];
                    const REAL ylm_im = coeff * plm * \
                        exp_vec[EXP_IM_IND(nlevel, mx)];
                    
                    REAL pe_im = 0.0;

                    cplx_mul_add(
                        cube_start[CUBE_IND(lx, mx)],
                        cube_start_im[CUBE_IND(lx, mx)],
                        ylm_re,
                        ylm_im,
                        &local_pe,
                        &pe_im
                    );
                    

                }
            }
            
        count++;
        potential_energy += local_pe*0.5;
        //potential_energy += local_pe;
    }   

    // printf("npart %d count %d\n", npart, count);

    if (err < 0){return err;}
    if (count != npart) {err = -4;}

    *phi_data = potential_energy;
    return err;
}




/*





extern "C"
INT32 particle_extraction(
    const INT64 nlevel,
    const UINT64 npart,
    const INT32 thread_max,
    const REAL * RESTRICT position,             // xyz
    const REAL * RESTRICT charge,
    const INT32 * RESTRICT fmm_cell,
    const REAL * RESTRICT boundary,             // xl. xu, yl, yu, zl, zu
    const UINT64 * RESTRICT cube_offset,        // zyx (slowest to fastest)
    const UINT64 * RESTRICT cube_dim,           // as above
    const UINT64 * RESTRICT cube_side_counts,   // as above
    const REAL * RESTRICT local_moments,
    REAL * RESTRICT phi_data,                  // lexicographic
    const INT32 * RESTRICT thread_assign
){
    INT32 err = 0;
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


    // Above we assigned particles to threads in a way that avoids needing atomics here.
    // Computing the cell is duplicate work, but the computation is cheap and the positions
    // needed to be read anyway to compute the moments.

    // using omp parallel for with schedule(static, 1) is robust against the omp
    // implementation deciding to launch less threads here than thread_max (which it
    // is entitled to do).
    
    REAL exp_space[thread_max][nlevel*4 + 2];
    // pre compute factorial and double factorial
    const UINT64 nfact = (2*nlevel > 4) ? 2*nlevel : 4;
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

    UINT32 count = 0;
    REAL potential_energy = 0.0;

    #pragma omp parallel for default(none) shared(thread_assign, position, \
        boundary, cube_offset, cube_dim, err, exp_space, \
        factorial_vec, double_factorial_vec, P_SPACE_VEC, \
        cube_half_side_len, cube_ilen, charge, local_moments) \
        schedule(static,1) \
        reduction(+: count) reduction(+: potential_energy)
    for(INT32 tx=0 ; tx<thread_max ; tx++){
        const int tid = omp_get_thread_num();
        const UINT64 ncomp = nlevel*nlevel*2;
        REAL * P_SPACE = P_SPACE_VEC[tid];
        REAL local_pe = 0.0;

        for(INT64 px=0 ; px<thread_assign[tx] ; px++){
            INT64 ix = thread_assign[thread_max + npart*tx + px];
            
            if (ix<0) {
                #pragma omp critical
                {err = -5;}
            } else if (ix >= npart) {
                #pragma omp critical
                {err = -6;}
            }


            REAL radius, ctheta, cphi, sphi, msphi;
            const INT64 ix_cell = compute_cell_spherical(
                cube_ilen, cube_half_side_len, position[ix*3], position[ix*3+1], 
                position[ix*3+2], boundary, cube_offset, cube_dim,
                &radius, &ctheta, &cphi, &sphi, &msphi
            );
            
            if (tx != ix_cell % thread_max) {           
                #pragma omp critical
                {err = -3;}
            }

            //compute spherical harmonic moments
            // start with the complex exponential (will not vectorise)
            REAL * RESTRICT exp_vec = exp_space[tid]; 
            const REAL * RESTRICT cube_start = &local_moments[ix_cell*ncomp];
            const REAL * RESTRICT cube_start_im = \
                &local_moments[ix_cell*ncomp + nlevel*nlevel];


            exp_vec[EXP_RE_IND(nlevel, 0)] = 1.0;
            exp_vec[EXP_IM_IND(nlevel, 0)] = 0.0;
            for (INT32 lx=1 ; lx<=((INT32)nlevel) ; lx++){
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
                    P_SPACE[P_SPACE_IND(nlevel, lx+1, lx+1)] = \ 
                    (-1.0 - 2.0*lx) * sqrt_1m2lx * \ 
                    P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

                    P_SPACE[P_SPACE_IND(nlevel, lx+1, lx)] = \
                        ctheta * (2*lx + 1) * \
                        P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

                    for( int mx=0 ; mx<lx ; mx++ ){
                        P_SPACE[P_SPACE_IND(nlevel, lx+1, mx)] = \
                            (ctheta * (2.0*lx+1.0) * \
                            P_SPACE[P_SPACE_IND(nlevel, lx, mx)] - \
                            (lx+mx)*P_SPACE[P_SPACE_IND(nlevel, lx-1, mx)])/ \
                            (lx - mx + 1);
                    }

                }
            }

            
            REAL rhol = 1.0;
            //loop over l and m
            for( int lx=0 ; lx<((int)nlevel) ; lx++ ){
                rhol = (lx > 0) ? rhol*radius : 1.0;

                for( int mx=-1*lx ; mx<=lx ; mx++ ){
                    const UINT32 abs_mx = abs(mx);
                    const REAL coeff = sqrt(factorial_vec[lx - abs_mx]/
                        factorial_vec[lx + abs_mx]) * charge[ix] * rhol;

                    const REAL plm = P_SPACE[P_SPACE_IND(nlevel, lx, abs_mx)];
                    const REAL ylm_re = coeff * plm * \
                        exp_vec[EXP_RE_IND(nlevel, mx)];
                    const REAL ylm_im = coeff * plm * \
                        exp_vec[EXP_IM_IND(nlevel, mx)];
                    
                    REAL pe_im = 0.0;

                    cplx_mul_add(
                        cube_start[CUBE_IND(lx, mx)],
                        cube_start_im[CUBE_IND(lx, mx)],
                        ylm_re,
                        ylm_im,
                        &local_pe,
                        &pe_im
                    );
                    

                }
            }
            
            count++;
        }
        potential_energy += local_pe*0.5;
        //potential_energy += local_pe;
    }   

    // printf("npart %d count %d\n", npart, count);

    if (err < 0){return err;}
    if (count != npart) {err = -4;}

    *phi_data = potential_energy;
    return err;
}








*/
