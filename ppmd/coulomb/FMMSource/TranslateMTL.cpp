

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

#define IPOW_RE(n) (1.0 - ((n)&2))
#define IPOW_IM(n) (0.0)


static inline void mtl(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT   ydata,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    REAL * RESTRICT         ldata
){

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 nlevel4 = nlevel*4;
    const INT64 im_offset = nlevel*nlevel;
    
    const INT64 nblk = 2*nlevel+2;
    REAL iradius_n[nblk];
    
    const REAL iradius = 1./radius;
    iradius_n[0] = 1.0;

    for(INT64 nx=1 ; nx<nblk ; nx++){
        iradius_n[nx] = iradius_n[nx-1] * iradius;
    }
    
    // loop over parent moments
    for(INT32 jx=0     ; jx<nlevel ; jx++ ){
    for(INT32 kx=-1*jx ; kx<=jx    ; kx++){
        const INT64 akx = abs(kx);
        const REAL ajk = a_array[jx * nlevel4 + kx];     // A_j^k
        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;
        

        for(INT32 nx=0     ; nx<=jx ; nx++){
            const REAL m1tn = 1.0 - 2.0*((REAL)(nx & 1));   // -1^{n}
            for(INT64 mx=-1*nx ; mx<=nx ; mx++){
                const INT64 abs_mx_kx = abs(mx - kx);

                const REAL anm = a_array[nx * nlevel4 + mx];    // A_n^m
                const REAL ra_jn_mk = ar_array[(jx + nx)*nlevel4 + abs_mx_kx];    // 1 / A_{j + n}^{m - k}
                const REAL rr_jn1 = iradius_n[jx + nx + 1];     // 1 / rho^{j + n + 1}
                
                //const INT64 amx = ((mx < 0) ? -1 : 1) * mx;
                const INT64 amx = abs(mx);

                const INT64 ip = abs_mx_kx - akx - amx;
                const REAL coeff_re = IPOW_RE(ip) * m1tn * anm * ajk * ra_jn_mk * rr_jn1;
                //const REAL coeff_re = ((REAL)(ip & 2))*m1tn * anm * ajk * ra_jn_mk * rr_jn1;
                //const REAL coeff_im1 = (ip < 0) ? -1.0: 1.0;
                //const REAL coeff_im = IPOW_IM(ip) * coeff_im1;
                const REAL coeff_im = 0.0;
                
                REAL ocoeff_re;
                REAL ocoeff_im;
                cplx_mul(       odata[CUBE_IND(nx, mx)], odata[CUBE_IND(nx, mx) + im_offset], 
                                coeff_re, coeff_im, &ocoeff_re, &ocoeff_im);

                cplx_mul_add(   ydata[CUBE_IND(jx+nx, mx-kx)], ydata[CUBE_IND(jx+nx, mx-kx) + im_offset],
                                ocoeff_re, ocoeff_im, &contrib_re, &contrib_im);
                
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
    const REAL * RESTRICT   ydata,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    REAL * RESTRICT         ldata
){
    mtl(      
    nlevel,
    radius,
    odata,
    ydata,
    a_array,
    ar_array,
    ldata
    );
    return 0;
}

