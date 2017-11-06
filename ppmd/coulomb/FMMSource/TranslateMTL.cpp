

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

//static const REAL ipow_re[4]     = {1.0, 0.0, -1.0, 0.0};
//static const REAL ipow_im[4]     = {0.0, 1.0, 0.0, -1.0};
//static const REAL ipow_im_neg[4] = {0.0, -1.0, 0.0, 1.0};
//#define IPOW_RE(n) (ipow_re[(n) & 3])
//#define IPOW_IM(n) (ipow_im[(n) & 3])


//#define IPOW_RE(n) ((1. - ((n)&1)) * (1. - ((n)&2)))
//#define IPOW_IM(n) (((n)&1)*(1.- ((n)&2)))

//static double factorial(const INT64 n) {
//    REAL 
//}

static inline void mtl(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT   phi_data,
    const REAL * RESTRICT   theta_data,
    const REAL * RESTRICT   theta_coeff,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    const REAL * RESTRICT   i_array,
    REAL * RESTRICT         ldata
){
    const INT64 ASTRIDE1 = 4*nlevel + 1;
    const INT64 ASTRIDE2 = 2*nlevel;

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 nlevel4 = nlevel*4;
    const INT64 im_offset = nlevel*nlevel;
    
    const INT64 nblk = 2*nlevel+2;
    REAL iradius_n[nblk];
    
    const REAL iradius = 1./radius;
    iradius_n[0] = 1.0;

    for(INT64 nx=1 ; nx<nblk ; nx++){ iradius_n[nx] = iradius_n[nx-1] * iradius; }

    REAL * RESTRICT iradius_p1 = &iradius_n[1];

    // loop over parent moments
    for(INT32 jx=0     ; jx<nlevel ; jx++ ){
    for(INT32 kx=-1*jx ; kx<=jx    ; kx++){
        const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];     // A_j^k
        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;

        for(INT32 nx=0     ; nx<nlevel ; nx++){
            const REAL m1tn = 1.0 - 2.0*((REAL)(nx & 1));   // -1^{n}
            const INT64 jxpnx = jx + nx;
            const INT64 p_ind_base = P_IND(jxpnx, 0);
            const REAL rr_jn1 = iradius_p1[jxpnx];     // 1 / rho^{j + n + 1}

            for(INT64 mx=-1*nx ; mx<=nx ; mx++){

                const INT64 mxmkx = mx - kx;

                // construct the spherical harmonic
                const INT64 y_aind = p_ind_base + mxmkx;
                const REAL y_coeff = theta_coeff[CUBE_IND(jx+nx,mx-kx)] *\
                    theta_data[CUBE_IND(jx+nx,mx-kx)];
                const REAL y_re = y_coeff * \
                    phi_data[EXP_RE_IND(2*nlevel, mxmkx)];
                const REAL y_im = y_coeff * \
                    phi_data[EXP_IM_IND(2*nlevel, mxmkx)];

                // compute translation coefficient
                // A_n^m
                const REAL anm = a_array[nx*ASTRIDE1 + ASTRIDE2 + mx];

                // 1 / A_{j + n}^{m - k}
                const REAL ra_jn_mk = \
                    ar_array[(jxpnx)*ASTRIDE1 + ASTRIDE2 + mxmkx];
                
                const REAL coeff_re = \
                    i_array[(nlevel+kx)*(nlevel*2 + 1) + nlevel + mx] * \
                    m1tn * anm * ajk * ra_jn_mk * rr_jn1;
                
                const INT64 oind = CUBE_IND(nx, mx);
                const REAL ocoeff_re = odata[oind]              * coeff_re;
                const REAL ocoeff_im = odata[oind + im_offset]  * coeff_re;

                cplx_mul_add(y_re, y_im, 
                    ocoeff_re, ocoeff_im, &contrib_re, &contrib_im);

                //if(jx == 0 && kx == 0){
                //    printf("nx\t%d\tmx\t%d:\t%f\t%f\n", 
                //        nx, mx, ocoeff_re, odata[oind]);
                //}

            }
        }
        
        ldata[CUBE_IND(jx, kx)] += contrib_re;
        ldata[CUBE_IND(jx, kx) + im_offset] += contrib_im;

    }}
}


extern "C"
int mtl_test_wrapper(  
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT   phi_data,
    const REAL * RESTRICT   theta_data,
    const REAL * RESTRICT   theta_coeff,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    const REAL * RESTRICT   i_array,
    REAL * RESTRICT         ldata
){
    mtl(      
    nlevel,
    radius,
    odata,
    phi_data,
    theta_data,
    theta_coeff,
    a_array,
    ar_array,
    i_array,
    ldata
    );
    return 0;
}



extern "C"
int test_i_power(
    const REAL * RESTRICT i_array,
    const INT64 nlevel,
    const INT64 kx,
    const INT64 mx,
    REAL * re_part
){
    *re_part = i_array[(nlevel+kx)*(nlevel*2 + 1) + nlevel + mx];
    return 0;
    
}

static inline void mtl_octal(
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
    
    const INT64 nblk = 2*nlevel+2;
    REAL iradius_n[nblk];
    
    const REAL iradius = 1./radius;
    iradius_n[0] = 1.0;

    for(INT64 nx=1 ; nx<nblk ; nx++){ 
        iradius_n[nx] = iradius_n[nx-1] * iradius;
    }

    REAL * RESTRICT iradius_p1 = &iradius_n[1];

    // loop over parent moments
    for(INT32 jx=0     ; jx<nlevel ; jx++ ){
    for(INT32 kx=-1*jx ; kx<=jx    ; kx++){
        // A_j^k
        const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];
        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;

        for(INT32 nx=0     ; nx<nlevel ; nx++){
            // -1^{n}
            const REAL m1tn = 1.0 - 2.0*((REAL)(nx & 1));
            const INT64 jxpnx = jx + nx;
            const INT64 p_ind_base = P_IND(jxpnx, 0);
            // 1 / rho^{j + n + 1}
            const REAL rr_jn1 = iradius_p1[jxpnx];     

            for(INT64 mx=-1*nx ; mx<=nx ; mx++){

                const INT64 mxmkx = mx - kx;

                // construct the spherical harmonic
                const REAL y_re = y_data[CUBE_IND(jx+nx, mx-kx)];
                const REAL y_im = y_data[im_offset2 + \
                    CUBE_IND(jx+nx, mx-kx)];

                // compute translation coefficient
                const REAL anm = a_array[
                    nx*ASTRIDE1 + ASTRIDE2 + mx];    // A_n^m
                // 1 / A_{j + n}^{m - k}
                const REAL ra_jn_mk = ar_array[(jxpnx)*ASTRIDE1 +\
                    ASTRIDE2 + mxmkx];
                
                const REAL coeff_re = i_array[(nlevel+kx)*(nlevel*2 + 1)+\
                    nlevel + mx] *\
                    m1tn * anm * ajk * ra_jn_mk * rr_jn1;
                
                const INT64 oind = CUBE_IND(nx, mx);
                const REAL ocoeff_re = odata[oind]              * coeff_re;
                const REAL ocoeff_im = odata[oind + im_offset]  * coeff_re;
                cplx_mul_add(   y_re, y_im, 
                                ocoeff_re, ocoeff_im, 
                                &contrib_re, &contrib_im);

                if(jx == 0 && kx == 0){
                    printf("nx\t%d\tmx\t%d:\t%f\t%f\n",
                    nx, mx, rr_jn1,  y_re);
                }

            }
        }
        
        ldata[CUBE_IND(jx, kx)] += contrib_re;
        ldata[CUBE_IND(jx, kx) + im_offset] += contrib_im;
        
    }}
    printf("---- %f\n", radius);
}



extern "C"
int translate_mtl_octal(
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

    //#pragma omp parallel for default(none) schedule(dynamic) \
    //shared(dim_parent, dim_child, moments_child, moments_parent, \
    //ylm, alm, almr, i_array)
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
            mtl_octal(nlevel, radius, pd_re, &ylm[childx*ncomp2], alm, almr, i_array, cd_re[childx]);
        }

    }

    return err;
}



