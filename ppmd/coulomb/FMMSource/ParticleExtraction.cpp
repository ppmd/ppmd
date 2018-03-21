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

static inline void cplx_mul(
    const REAL a,
    const REAL b,
    const REAL x,
    const REAL y,
    REAL * RESTRICT g,
    REAL * RESTRICT h
){
   // ( a + bi) * (x + yi) = (ax - by) + (xb + ay)i
    *g = a * x - b * y;
    *h = x * b + a * y;
}




static inline INT64 get_cube_index(
    const INT64 * RESTRICT cube_offset,
    const INT64 * RESTRICT cube_dim,
    const INT64 * RESTRICT cube_side_counts,   
    const INT64 global_cell
){
    
    const INT64 nsx = cube_side_counts[2];
    const INT64 nsy = cube_side_counts[1];
    const INT64 nsz = cube_side_counts[0];

    const INT64 cxt = global_cell % nsx;
    const INT64 cyt = ((global_cell - cxt) / nsx) % nsy;
    const INT64 czt = (global_cell - cxt - cyt*nsx) / (nsx*nsy);

    const INT64 cx = cxt - cube_offset[2];
    const INT64 cy = cyt - cube_offset[1];
    const INT64 cz = czt - cube_offset[0];

    if (cx >= cube_dim[2] || cy >= cube_dim[1] || cz >= cube_dim[0] ){
        return (INT64) -1;}

    return cx + cube_dim[2] * (cy + cube_dim[1] * cz);;
}


static inline INT64 get_cube_midpoint(
    const REAL * RESTRICT cube_half_len,
    const REAL * RESTRICT boundary,
    const INT64 * RESTRICT cube_offset,
    const INT64 * RESTRICT cube_dim,
    const INT64 * RESTRICT cube_side_counts,   
    const INT64 global_cell,
    REAL * RESTRICT midx,
    REAL * RESTRICT midy,
    REAL * RESTRICT midz
){
    
    const INT64 nsx = cube_side_counts[2];
    const INT64 nsy = cube_side_counts[1];
    const INT64 nsz = cube_side_counts[0];

    const INT64 cxt = global_cell % nsx;
    const INT64 cyt = ((global_cell - cxt) / nsx) % nsy;
    const INT64 czt = (global_cell - cxt - cyt*nsx) / (nsx*nsy);

    const INT64 cx = cxt - cube_offset[2];
    const INT64 cy = cyt - cube_offset[1];
    const INT64 cz = czt - cube_offset[0];


    if (cx >= cube_dim[2] || cy >= cube_dim[1] || cz >= cube_dim[0] ){
        return (INT64) -1;}
    //const int cell_idx = cx + cube_dim[2] * (cy + cube_dim[1] * cz);
    // now compute the vector between particle and cell center in spherical coords
    // cell center
    *midx = (cxt * 2 + 1) * cube_half_len[0] - 0.5 * boundary[0]; 
    *midy = (cyt * 2 + 1) * cube_half_len[1] - 0.5 * boundary[1]; 
    *midz = (czt * 2 + 1) * cube_half_len[2] - 0.5 * boundary[2]; 
    return 0;
}


static inline INT64 get_offset_vector(
    const REAL ccx,
    const REAL ccy,
    const REAL ccz,
    const REAL px,
    const REAL py,
    const REAL pz,
    REAL * RESTRICT radius,
    REAL * RESTRICT ctheta,
    REAL * RESTRICT stheta,
    REAL * RESTRICT cphi,
    REAL * RESTRICT sphi,
    REAL * RESTRICT msphi
){
    
    // compute Cartesian displacement vector
    const REAL dx = px - ccx;
    const REAL dy = py - ccy;
    const REAL dz = pz - ccz;

    // convert to spherical
    const REAL dx2 = dx*dx;
    const REAL dx2_p_dy2 = dx2 + dy*dy;
    const REAL d2 = dx2_p_dy2 + dz*dz;
    *radius = sqrt(d2);
    
    const REAL theta = atan2(sqrt(dx2_p_dy2), dz);
    *ctheta = cos(theta);
    *stheta = sin(theta);
    
    const REAL phi = atan2(dy, dx);

    *cphi = cos(phi);
    *sphi = sin(phi); 
    *msphi = -1.0 * (*sphi);
    return 0;
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


static void compute_p_space(
    const int nlevel, 
    const REAL x, 
    REAL * RESTRICT P_SPACE
){
    const REAL sqrt_1m2lx = sqrt(1.0 - x*x);

    // P_0^0 = 1;
    P_SPACE[P_SPACE_IND(nlevel, 0, 0)] = 1.0;
    for( int lx=0 ; lx<((int)nlevel) ; lx++){

        //compute the (lx+1)th P values using the lx-th values
        if (lx<(nlevel-1) ){ 
            P_SPACE[P_SPACE_IND(nlevel, lx+1, lx+1)] = \ 
            (-1.0 - 2.0*lx) * sqrt_1m2lx * \ 
            P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

            P_SPACE[P_SPACE_IND(nlevel, lx+1, lx)] = \
                x * (2*lx + 1) * \
                P_SPACE[P_SPACE_IND(nlevel, lx, lx)];

            for( int mx=0 ; mx<lx ; mx++ ){
                P_SPACE[P_SPACE_IND(nlevel, lx+1, mx)] = \
                    (x * (2.0*lx+1.0) * \
                    P_SPACE[P_SPACE_IND(nlevel, lx, mx)] - \
                    (lx+mx)*P_SPACE[P_SPACE_IND(nlevel, lx-1, mx)])/ \
                    (lx - mx + 1);

            }
        }
    }
}

static inline void compute_exp_space(
    const int nlevel,
    const REAL cphi,
    const REAL sphi,
    const REAL msphi,
    REAL * exp_vec
){
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
}


/*
The expression to compute the gradient of the potential field is badly defined
if a particle is positioned on a coordinate axis relative the expansion point.
We avoid a badly defined expression (and doing some analysis) by shifting the
local expansion to a place where the gradient is well defined.
*/
static inline void ltl(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    const REAL * RESTRICT   i_array,
    const REAL * RESTRICT   factorial_vec,
    const REAL * RESTRICT   P_SPACE,
    const REAL * RESTRICT   exp_vec,
    REAL * RESTRICT         ldata
){
    const INT64 ASTRIDE1 = 4*nlevel + 1;
    const INT64 ASTRIDE2 = 2*nlevel;

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 ncomp2 = nlevel*nlevel*8;
    
    const INT64 nlevel4 = nlevel*4;
    const INT64 im_offset = nlevel*nlevel;
    const INT64 im_offset2 = nlevel*nlevel*4;
    
    const INT64 nblk = nlevel+1;
    REAL radius_n[nblk];
    
    radius_n[0] = 1.0;

    for(INT64 nx=1 ; nx<nblk ; nx++){ 
        radius_n[nx] = radius_n[nx-1] * radius;
    }

    // loop over parent moments
    for(INT64 jx=0     ; jx<nlevel ; jx++ ){
    for(INT64 kx=-1*jx ; kx<=jx    ; kx++){
        // A_j^k
        const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];

        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;

        for(INT64 nx=jx ; nx<nlevel ; nx++){
            // -1^{n}
            const REAL m1tnpj = 1.0 - 2.0*((REAL)((nx+jx) & 1));

            const INT64 jxpnx = jx + nx;

            // 1 / rho^{j + n + 1}
            const REAL r_n_j = radius_n[nx-jx];

            for(INT64 mx=-1*nx ; mx<=nx ; mx++){

                const INT64 mxmkx = mx - kx;
                const INT64 nxmjx = nx - jx;

                const bool valid_indices = ABS(mxmkx) <= ABS(nxmjx);

                // construct the spherical harmonic

                const REAL ycoeff = valid_indices ? sqrt(factorial_vec[nxmjx - ABS(mxmkx)]/
                    factorial_vec[nxmjx + ABS(mxmkx)]) : 0.0;

                const REAL plm =  valid_indices ? P_SPACE[P_SPACE_IND(nlevel, nxmjx, ABS(mxmkx))]: 0.0;
                const REAL y_re = valid_indices ? ycoeff * plm * exp_vec[EXP_RE_IND(nlevel, mxmkx)] : 0.0;
                const REAL y_im = valid_indices ? ycoeff * plm * exp_vec[EXP_IM_IND(nlevel, mxmkx)] : 0.0;

                // A_n^m
                const REAL a_nj_mk = a_array[
                    (nx-jx)*ASTRIDE1 + ASTRIDE2 + (mxmkx)];

                // 1 / A_{j + n}^{m - k}
                const REAL ra_n_m = ar_array[nx*ASTRIDE1 +\
                    ASTRIDE2 + mx];

                const REAL coeff_re = i_array[(nlevel+kx)*(nlevel*2 + 1)+\
                    nlevel + mx] *\
                    m1tnpj * a_nj_mk * ajk * ra_n_m * r_n_j;

                const INT64 oind = CUBE_IND(nx, mx);
                const REAL ocoeff_re = odata[oind]              * coeff_re;
                
                const REAL ocoeff_im = odata[oind + im_offset]  * coeff_re;

                cplx_mul_add(   y_re, y_im, 
                                ocoeff_re, ocoeff_im, 
                                &contrib_re, &contrib_im);

            }
        }


        ldata[CUBE_IND(jx, kx)] = contrib_re;
        ldata[CUBE_IND(jx, kx) + im_offset] = contrib_im;
        
    }}
}




extern "C"
INT64 particle_extraction(
    const INT64 nlevel,
    const INT64 npart,
    const INT64 thread_max,
    const REAL * RESTRICT position,             // xyz
    const REAL * RESTRICT charge,
    REAL * RESTRICT force,
    const INT64 * RESTRICT fmm_cell,
    const REAL * RESTRICT boundary,             // xl. xu, yl, yu, zl, zu
    const INT64 * RESTRICT cube_offset,        // zyx (slowest to fastest)
    const INT64 * RESTRICT cube_dim,           // as above
    const INT64 * RESTRICT cube_side_counts,   // as above
    REAL * RESTRICT local_moments,
    REAL * RESTRICT phi_data,                  // lexicographic
    const INT64 * RESTRICT thread_assign,
    const REAL * RESTRICT alm,
    const REAL * RESTRICT almr,
    const REAL * RESTRICT i_array,
    const INT64 always_shift,
    INT64 * RESTRICT shift_yes,
    INT64 * RESTRICT shift_no
){
    INT64 err = 0;
    omp_set_num_threads(thread_max);


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

    
    REAL exp_space[thread_max][nlevel*4 + 2];
    // pre compute factorial and double factorial
    const INT64 nfact = (2*nlevel > 4) ? 2*nlevel : 4;
    REAL factorial_vec[nfact];

    factorial_vec[0] = 1.0;
    factorial_vec[1] = 1.0;
    factorial_vec[2] = 2.0;

    for( INT64 lx=3 ; lx<nfact ; lx++ ){
        factorial_vec[lx] = lx * factorial_vec[lx-1];
    }
    
    REAL P_SPACE_VEC[thread_max][nlevel*nlevel*2];
    REAL L_SPACE_VEC[thread_max][nlevel*nlevel*2];

    INT64 count = 0;
    INT64 _shift_yes = 0;
    INT64 _shift_no = 0;

    REAL potential_energy = 0.0;
    #pragma omp parallel for default(none) shared(thread_assign, position, \
        boundary, cube_offset, cube_dim, err, exp_space, force, \
        factorial_vec, P_SPACE_VEC, L_SPACE_VEC,\
        cube_half_side_len, cube_ilen, charge, local_moments, fmm_cell, cube_side_counts,\
        alm, almr, i_array) \
        schedule(dynamic) \
        reduction(+: count) reduction(+: potential_energy) reduction(+:_shift_yes) reduction(+:_shift_no)
    for(INT64 ix=0 ; ix<npart ; ix++){
        const int tid = omp_get_thread_num();
        
        //printf("particle: %d\n", ix);

        const INT64 ncomp = nlevel*nlevel*2;

        // threads tmp space for exponentials and legendre polynomials
        REAL * RESTRICT P_SPACE = P_SPACE_VEC[tid];
        REAL * RESTRICT exp_vec = exp_space[tid]; 
        //REAL * RESTRICT L_SPACE = L_SPACE_VEC[tid];

        for(int tx=0 ; tx<nlevel*nlevel*2 ; tx++ ){ P_SPACE[tx]=12345.678; }
        for(int tx=0 ; tx<(nlevel*4 + 2) ; tx++ ){ exp_vec[tx]=12345.678; }
        
        // the fmm cell a particle is contained in is determined once in the initial 
        // binning into cells.
        const INT64 ix_cell = get_cube_index(cube_offset,cube_dim, cube_side_counts, fmm_cell[ix]);


        REAL * RESTRICT cube_start = &local_moments[ix_cell*ncomp];

        // cell mid points
        REAL midx, midy, midz;
        // spherical co-ordinate displacement vector
        REAL radius, ctheta, stheta, cphi, sphi, msphi;
        
        
        get_cube_midpoint(cube_half_side_len, boundary, cube_offset, cube_dim,
            cube_side_counts, fmm_cell[ix], &midx, &midy, &midz);

        const REAL px = position[ix*3];
        const REAL py = position[ix*3+1];
        const REAL pz = position[ix*3+2];
        
        const REAL tol = 0.00000000001;

        const bool shx = ABS(px-midx) < tol;
        const bool shy = ABS(py-midy) < tol;
        const bool shz = ABS(pz-midz) < tol;
        
        // see comment above ltl function.
///*        
        bool shift_expansion = (
            ((always_shift == 1) || shx || shy || shz) && (!(always_shift == -1))
            ) ? true : false;
//*/        
//shift_expansion = false;
//


        get_offset_vector(midx, midy, midz, px, py, pz,
            &radius, &ctheta, &stheta, &cphi, &sphi, &msphi);
        
        shift_expansion = (
            ((always_shift == 1) || (radius < tol) || (stheta<tol)) && (!(always_shift == -1))
            ) ? true : false;




//printf("SHIFTING IS DISABLED!\n");
        REAL * RESTRICT L_SPACE;

        if (shift_expansion){
            
            L_SPACE = L_SPACE_VEC[tid];
            
            // can shift local expansion to well defined place here
            REAL newx = px + 0.1*cube_half_side_len[2];
            REAL newy = py + 0.1*cube_half_side_len[1];
            REAL newz = pz + 0.1*cube_half_side_len[0];
            

            // displacement from new expansion point to cube centre.
            get_offset_vector(newx, newy, newz, midx, midy, midz,
                &radius, &ctheta, &stheta, &cphi, &sphi, &msphi);        
            compute_p_space(nlevel, ctheta, P_SPACE);
            compute_exp_space(nlevel, cphi, sphi, msphi, exp_vec);

            ltl(nlevel, radius, cube_start, alm, almr, i_array,
                factorial_vec, P_SPACE, exp_vec, L_SPACE);

            get_offset_vector(newx, newy, newz, px, py, pz,
                &radius, &ctheta, &stheta, &cphi, &sphi, &msphi);

            compute_p_space(nlevel, ctheta, P_SPACE);
            compute_exp_space(nlevel, cphi, sphi, msphi, exp_vec);

            _shift_yes++;
        } else {

            L_SPACE = cube_start;

            // compute offset vectors between particle and point of expansion
            get_offset_vector(midx, midy, midz, px, py, pz,
                &radius, &ctheta, &stheta, &cphi, &sphi, &msphi);
            compute_p_space(nlevel, ctheta, P_SPACE);
            compute_exp_space((int)nlevel, cphi, sphi, msphi, exp_vec);

            _shift_no++;
        }

        const REAL _rstheta = 1.0/stheta;
        const REAL rstheta = (std::isnan(_rstheta) || std::isinf(_rstheta)) ? 0.0 : _rstheta;
        //const REAL rstheta = _rstheta;

        REAL local_pe  = 0.0;
        REAL sp_radius = 0.0;
        REAL sp_phi    = 0.0;
        REAL sp_theta  = 0.0;

        // compute potential energy contribution
        REAL rhol = 1.0;
        REAL rhol2 = 1.0/radius;
        //loop over l and m


//printf("radius %f, ctheta %f, stheta %f, cphi %f, sphi%f \n" , radius, ctheta, stheta, cphi, sphi);
//printf("LOOP IS MODDED!\n");
        for( int lx=0 ; lx<((int)nlevel) ; lx++ ){
//for( int lx=0 ; lx<2 ; lx++ ){
            
            if (lx==0 && (std::isnan(rhol2) || std::isinf(rhol2))) {rhol2 = 0.0;}
            if (lx==1) {rhol2 = 1.0;}


            for( int mx=-1*lx ; mx<=lx ; mx++ ){
                // energy computation
                const INT64 abs_mx = abs(mx);

                const REAL ycoeff = sqrt(factorial_vec[lx - abs_mx]/
                    factorial_vec[lx + abs_mx]);

                const REAL plm = P_SPACE[P_SPACE_IND(nlevel, lx, abs_mx)];

                const REAL ylm_re = ycoeff * plm * \
                    exp_vec[EXP_RE_IND(nlevel, mx)];
                const REAL ylm_im = ycoeff * plm * \
                    exp_vec[EXP_IM_IND(nlevel, mx)];

                const REAL ljk_re = L_SPACE[CUBE_IND(lx, mx)];
                const REAL ljk_im = L_SPACE[CUBE_IND(lx, mx) + nlevel*nlevel];
                
                REAL pe_im = 0.0;
                cplx_mul_add(
                    ljk_re,
                    ljk_im,
                    ylm_re * rhol,
                    ylm_im * rhol,
                    &local_pe,
                    &pe_im
                );
                


                // force computation
                // radius part
                REAL tmp_re;
                REAL tmp_im;
                
                cplx_mul(ljk_re, ljk_im,
                         ylm_re * rhol2, ylm_im * rhol2,
                         &tmp_re, &tmp_im);

                sp_radius += tmp_re * ((REAL) lx);
                
                // phi part
                const REAL phi_coeff = rhol2 * ycoeff * plm * rstheta;
                
                const REAL phi_exp_re = -1 * ((REAL) mx) * exp_vec[EXP_IM_IND(nlevel, mx)];
                const REAL phi_exp_im =      ((REAL) mx) * exp_vec[EXP_RE_IND(nlevel, mx)];

                cplx_mul(ljk_re, ljk_im,
                         phi_exp_re, phi_exp_im,
                         &tmp_re, &tmp_im);
                
                sp_phi += phi_coeff * tmp_re;
    
                // theta part
                // P_{-j}^k = P_{j-1}^k => P_{-1}^k = P_0_k = 1.0 if k==0, 0 o/w.
                const REAL plmm1_1 = (lx==0) ? ( (mx==0) ? 1.0 : 0.0 ) : P_SPACE[P_SPACE_IND(nlevel, lx-1, abs_mx)];
                const REAL plmm1 = plmm1_1 * ( (ABS(lx-1) < abs_mx ) ? 0.0 : 1.0);
                
//printf("j= %d, k= %d, P= %f | cos(theta) %f \n", lx, mx, plmm1, ctheta);


//if (ABS(12345.678 - plmm1)<0.0001) {printf("BAD plmm1 %d %d \n", lx, mx);}
//PRINT_NAN(plmm1)
                const REAL theta_coeff = -1.0 * rhol2 * ycoeff * rstheta * (
                    (((REAL) lx) +  abs_mx ) * plmm1  - ((REAL) lx ) * ctheta * plm
                );
                
                const REAL theta_exp_re = exp_vec[EXP_RE_IND(nlevel, mx)];
                const REAL theta_exp_im = exp_vec[EXP_IM_IND(nlevel, mx)];
                
//if ((EXP_RE_IND(nlevel, mx) < 0) || (EXP_RE_IND(nlevel, mx) >= nlevel*4 + 2)) {printf("bad real exp index");}
//if ((EXP_IM_IND(nlevel, mx) < 0) || (EXP_IM_IND(nlevel, mx) >= nlevel*4 + 2)) {printf("bad imag exp index");}
//if (ABS(12345.678 - theta_exp_re)<0.01) {printf("BAD theta_exp_re %d %d \n", lx, mx);}
//if (ABS(12345.678 - theta_exp_im)<0.01) {printf("BAD theta_exp_im %d %d \n", lx, mx);}

                cplx_mul(ljk_re, ljk_im,
                         theta_exp_re, theta_exp_im,
                         &tmp_re, &tmp_im);

//PRINT_NAN(theta_coeff)

                sp_theta += theta_coeff * tmp_re;


            }

            rhol *= radius;
            rhol2 *= radius;
        }
        potential_energy += local_pe*0.5*charge[ix];
        


        sp_radius *= charge[ix];
        sp_theta *= charge[ix];
        sp_phi *= charge[ix];
        
/*

Using the following unit vectors:

\hat{\vec{r}} = [ cos(phi) sin(theta), 
                  sin(phi)sin(theta), 
                  cos(theta)  ]

\hat{\vec{theta}} = [   cos(phi)cos(theta),
                        sin(phi)cos(theta),
                        -sin(theta) ]
 
\hat{\vec{phi}} = [     -sin(phi),
                        cos(phi),
                        0           ]
*/


PRINT_NAN(sp_radius)
PRINT_NAN(sp_theta)
PRINT_NAN(sp_phi)
PRINT_NAN(potential_energy)
        
//printf("sp_radius %f\n", sp_radius);
//printf("sp_theta  %f\n", sp_theta );
//printf("sp_phi    %f\n", sp_phi   );

        //printf("%f\n", sp_radius * cphi * stheta    +   sp_theta * cphi * ctheta    -   sp_phi * sphi);


        force[ix*3    ] -= sp_radius * cphi * stheta    +   sp_theta * cphi * ctheta    -   sp_phi * sphi;
        force[ix*3 + 1] -= sp_radius * sphi * stheta    +   sp_theta * sphi * ctheta    +   sp_phi * cphi;
        force[ix*3 + 2] -= sp_radius * ctheta           -   sp_theta * stheta;

        if(isbad(sp_radius) || isbad(sp_theta) || isbad(sp_phi) || isbad(potential_energy)){
            printf("ERROR: bad value detected (err code -5).\n");
            #pragma omp critical
            {err = -5;}
            #pragma omp flush (err)
        }


        count++;
    }   


    if (err < 0){return err;}
    if (count != npart) {err = -4;}

    *phi_data = potential_energy;

    *shift_yes += _shift_yes;
    *shift_no += _shift_no;
    return err;
}




