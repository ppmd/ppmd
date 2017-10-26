

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

    for(INT64 nx=1 ; nx<nblk ; nx++){
        iradius_n[nx] = iradius_n[nx-1] * iradius;
        printf("%f\n", iradius_n[nx]);
    }

    REAL * RESTRICT iradius_p1 = &iradius_n[1];
    

    // loop over parent moments
    for(INT32 jx=0     ; jx<nlevel ; jx++ ){
    for(INT32 kx=-1*jx ; kx<=jx    ; kx++){
        const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];     // A_j^k
        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;

        for(INT32 nx=0     ; nx<=jx ; nx++){
            const REAL m1tn = 1.0 - 2.0*((REAL)(nx & 1));   // -1^{n}
            const INT64 jxpnx = jx + nx;
            const INT64 p_ind_base = P_IND(jxpnx, 0);
            const REAL rr_jn1 = iradius_p1[jxpnx];     // 1 / rho^{j + n + 1}

            for(INT64 mx=-1*nx ; mx<=nx ; mx++){

                const INT64 mxmkx = mx - kx;

                // construct the spherical harmonic
                const INT64 y_aind = p_ind_base + mxmkx;
                const REAL y_coeff = theta_coeff[CUBE_IND(jx+nx,mx-kx)] * theta_data[CUBE_IND(jx+nx,mx-kx)];
                const REAL y_re = y_coeff * phi_data[EXP_RE_IND(2*nlevel, mxmkx)];
                const REAL y_im = y_coeff * phi_data[EXP_IM_IND(2*nlevel, mxmkx)];
                // compute translation coefficient
                const REAL anm = a_array[nx*ASTRIDE1 + ASTRIDE2 + mx];    // A_n^m
                const REAL ra_jn_mk = ar_array[(jxpnx)*ASTRIDE1 + ASTRIDE2 + mxmkx];    // 1 / A_{j + n}^{m - k}
                
                const REAL coeff_re = i_array[(nlevel+kx)*(nlevel*2 + 1) + nlevel + mx] *\
                    m1tn * anm * ajk * ra_jn_mk * rr_jn1;
                
                const INT64 oind = CUBE_IND(nx, mx);
                const REAL ocoeff_re = odata[oind]              * coeff_re;
                const REAL ocoeff_im = odata[oind + im_offset]  * coeff_re;

                printf("%.16f\n", rr_jn1);
                cplx_mul_add(y_re, y_im, ocoeff_re, ocoeff_im, &contrib_re, &contrib_im);
                
            }
        }
        
        ldata[CUBE_IND(jx, jx)] += contrib_re;
        ldata[CUBE_IND(jx, jx) + im_offset] += contrib_im;

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



