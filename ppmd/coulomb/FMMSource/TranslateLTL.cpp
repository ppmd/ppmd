

static inline void lin_to_xyz(
    const UINT32 * RESTRICT dim_parent,
    const INT64 lin,
    INT64 * RESTRICT cx,
    INT64 * RESTRICT cy,
    INT64 * RESTRICT cz
){
    *cx = lin % dim_parent[2];
    const INT64 yz = (lin - (*cx))/dim_parent[2];
    *cy = yz % dim_parent[1];
    *cz = (yz - (*cy))/dim_parent[1];
}

static inline INT64 xyz_to_lin(
    const UINT32 * RESTRICT dim_child,
    const INT64 cx,
    const INT64 cy,
    const INT64 cz
){
    return cx + dim_child[2]*(cy + dim_child[1]*cz);
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

static inline void ltl_octal(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT   y_data,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    const REAL * RESTRICT   i_array,
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
    for(INT32 jx=0     ; jx<nlevel ; jx++ ){
    for(INT32 kx=-1*jx ; kx<=jx    ; kx++){
        // A_j^k
        const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];
        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;

        for(INT32 nx=jx ; nx<nlevel ; nx++){
            // -1^{n}
            const REAL m1tnpj = 1.0 - 2.0*((REAL)((nx+jx) & 1));
            const INT64 jxpnx = jx + nx;
            const INT64 p_ind_base = P_IND(jxpnx, 0);
            // 1 / rho^{j + n + 1}
            const REAL r_n_j = radius_n[nx-jx];

            for(INT64 mx=-1*nx ; mx<=nx ; mx++){

                const INT64 mxmkx = mx - kx;

                // construct the spherical harmonic
                const REAL y_re = y_data[CUBE_IND(nx-jx, mx-kx)];
                const REAL y_im = y_data[im_offset2 + \
                    CUBE_IND(nx-jx, mx-kx)];

                // A_n^m
                const REAL a_nj_mk = a_array[
                    (nx-jx)*ASTRIDE1 + ASTRIDE2 + (mx-kx)];
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

                //if(jx == 0 && kx == 0){
                //    printf("nx\t%d\tmx\t%d:\t%f\t%f\n",
                //    nx, mx, rr_jn1,  y_re);
                //}

            }
        }
        
        ldata[CUBE_IND(jx, kx)] += contrib_re;
        ldata[CUBE_IND(jx, kx) + im_offset] += contrib_im;
        
    }}
    //printf("---- %f\n", radius);
}



extern "C"
int translate_ltl(
    const UINT32 * RESTRICT dim_parent,     // slowest to fastest
    const UINT32 * RESTRICT dim_child,      // slowest to fastest
    REAL * RESTRICT moments_child,
    const REAL * RESTRICT moments_parent,
    const REAL * RESTRICT ylm,
    const REAL * RESTRICT alm,
    const REAL * RESTRICT almr,
    const REAL * RESTRICT i_array,
    const REAL radius,
    const INT64 nlevel
){
    int err = 0;
    const INT64 nparent_cells = dim_parent[0] * dim_parent[1] * dim_parent[2];

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 ncomp2 = nlevel*nlevel*8;
    const INT64 im_offset = nlevel*nlevel;
    const INT64 im_offset2 = 4*nlevel*nlevel;

    #pragma omp parallel for default(none) schedule(dynamic) \
    shared(dim_parent, dim_child, moments_child, moments_parent, \
    ylm, alm, almr, i_array)
    for( INT64 pcx=0 ; pcx<nparent_cells ; pcx++ ){
        INT64 cx, cy, cz;
        lin_to_xyz(dim_parent, pcx, &cx, &cy, &cz);

        const REAL * RESTRICT pd_re = &moments_parent[pcx*ncomp];

        // child layer is type plain
        const INT64 ccx = 2*cx;
        const INT64 ccy = 2*cy;
        const INT64 ccz = 2*cz;

        //children are labeled lexicographically
        const INT64 cc0 = ncomp * xyz_to_lin(dim_child, ccx, ccy, ccz);
        const INT64 cc1 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy, ccz);
        const INT64 cc2 = ncomp * xyz_to_lin(dim_child, ccx, ccy+1, ccz);
        const INT64 cc3 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy+1, ccz);
        const INT64 cc4 = ncomp * xyz_to_lin(dim_child, ccx, ccy, ccz+1);
        const INT64 cc5 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy, ccz+1);
        const INT64 cc6 = ncomp * xyz_to_lin(dim_child, ccx, ccy+1, ccz+1);
        const INT64 cc7 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy+1, ccz+1);

        REAL * RESTRICT cd_re[8] = {
            &moments_child[cc7],
            &moments_child[cc6],
            &moments_child[cc5],
            &moments_child[cc4],
            &moments_child[cc3],
            &moments_child[cc2],
            &moments_child[cc1],
            &moments_child[cc0]
        };

        for(INT32 childx=0 ; childx<8 ; childx++ ){
            ltl_octal(nlevel, radius, pd_re, &ylm[childx*ncomp2], alm, almr, i_array, cd_re[childx]);
            //for(int tx=0 ; tx<4 ; tx++ ){
            //    printf("%d %f\n", tx, ylm[childx*ncomp2 + tx]);
            //}
        }

    }

    return err;
}



