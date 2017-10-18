

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

extern "C"
int translate_mtm(
    const UINT32 * RESTRICT dim_parent,     // slowest to fastest
    const UINT32 * RESTRICT dim_child,      // slowest to fastest
    const REAL * RESTRICT moments_child,
    REAL * RESTRICT moments_parent,
    const REAL * RESTRICT ylm,
    const REAL * RESTRICT alm,
    const REAL * RESTRICT almr,
    const REAL radius,
    const INT64 nlevel
){
    int err = 0;
    const INT64 nparent_cells = dim_parent[0] * dim_parent[1] * dim_parent[2];
    //loop over parent cells and pull data in from children
    

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 im_offset = nlevel*nlevel;

    const REAL ipow_re[4] = {1.0, 0.0, -1.0, 0.0};
    const REAL ipow_im[4] = {0.0, 1.0, 0.0, -1.0};
    #define IPOW_RE(n) (ipow_re[(n) & 3])
    #define IPOW_IM(n) (ipow_im[(n) & 3])
    
    REAL radius_n[nlevel];
    radius_n[0] = 1.0;
    for(INT64 nx=1 ; nx<nlevel ; nx++){
        radius_n[nx] = radius_n[nx-1] * radius;
    } 


    #pragma omp parallel for default(none) schedule(dynamic) shared(dim_parent,\
    dim_child,moments_child,moments_parent,ylm,alm,almr, radius_n, ipow_re, ipow_im)
    for( INT64 pcx=0 ; pcx<nparent_cells ; pcx++ ){
        INT64 cx, cy, cz;
        lin_to_xyz(dim_parent, pcx, &cx, &cy, &cz);
        
        REAL * RESTRICT pd_re = &moments_parent[pcx*ncomp];
        REAL * RESTRICT pd_im = &moments_parent[pcx*ncomp + im_offset];

        // child layer has halos
        const INT64 ccx = 2*cx + 2;
        const INT64 ccy = 2*cy + 2;
        const INT64 ccz = 2*cz + 2;

        //children are labeled lexicographically
        const INT64 cc0 = xyz_to_lin(dim_child, ccx, ccy, ccz);
        const INT64 cc1 = xyz_to_lin(dim_child, ccx+1, ccy, ccz);
        const INT64 cc2 = xyz_to_lin(dim_child, ccx, ccy+1, ccz);
        const INT64 cc3 = xyz_to_lin(dim_child, ccx+1, ccy+1, ccz);
        const INT64 cc4 = xyz_to_lin(dim_child, ccx, ccy, ccz+1);
        const INT64 cc5 = xyz_to_lin(dim_child, ccx+1, ccy, ccz+1);
        const INT64 cc6 = xyz_to_lin(dim_child, ccx, ccy+1, ccz+1);
        const INT64 cc7 = xyz_to_lin(dim_child, ccx+1, ccy+1, ccz+1);
        
        const REAL * RESTRICT cd0_re = &moments_child[cc0*ncomp];
        const REAL * RESTRICT cd1_re = &moments_child[cc1*ncomp];
        const REAL * RESTRICT cd2_re = &moments_child[cc2*ncomp];
        const REAL * RESTRICT cd3_re = &moments_child[cc3*ncomp];
        const REAL * RESTRICT cd4_re = &moments_child[cc4*ncomp];
        const REAL * RESTRICT cd5_re = &moments_child[cc5*ncomp];
        const REAL * RESTRICT cd6_re = &moments_child[cc6*ncomp];
        const REAL * RESTRICT cd7_re = &moments_child[cc7*ncomp];

        const REAL * RESTRICT cd0_im = &moments_child[cc0*ncomp + im_offset];
        const REAL * RESTRICT cd1_im = &moments_child[cc1*ncomp + im_offset];
        const REAL * RESTRICT cd2_im = &moments_child[cc2*ncomp + im_offset];
        const REAL * RESTRICT cd3_im = &moments_child[cc3*ncomp + im_offset];
        const REAL * RESTRICT cd4_im = &moments_child[cc4*ncomp + im_offset];
        const REAL * RESTRICT cd5_im = &moments_child[cc5*ncomp + im_offset];
        const REAL * RESTRICT cd6_im = &moments_child[cc6*ncomp + im_offset];
        const REAL * RESTRICT cd7_im = &moments_child[cc7*ncomp + im_offset];

        
        // loop over parent moments
        for(INT64 jx=0     ; jx<nlevel ; jx++ ){
        for(INT64 kx=-1*jx ; kx<=jx    ; kx++){
                const REAL ajkr = almr[jx*4*nlevel + abs(kx)];
                for(INT64 nx=0     ; nx<=jx ; nx++){
                for(INT64 mx=-1*nx ; mx<=nx ; mx++){

                    const REAL coeff = ajkr * radius_n[nx] * alm[nx*4*nlevel + abs(mx)] * \
                                       alm[(jx - nx)*4*nlevel + abs(kx - mx)];


                    const INT64 child_ind = CUBE_IND(jx - nx, kx - mx);
                    const INT64 child_ind_im = CUBE_IND(jx - nx, kx - mx) + im_offset;

                    const INT64 ychild_ind = CUBE_IND(nx, -1*mx);
                    const INT64 ychild_ind_im = CUBE_IND(nx, -1*mx) + im_offset;


                    const REAL child_re = (\
                        cd0_re[child_ind] * ylm[0*ncomp + ychild_ind] + \
                        cd1_re[child_ind] * ylm[1*ncomp + ychild_ind] + \
                        cd2_re[child_ind] * ylm[2*ncomp + ychild_ind] + \
                        cd3_re[child_ind] * ylm[3*ncomp + ychild_ind] + \
                        cd4_re[child_ind] * ylm[4*ncomp + ychild_ind] + \
                        cd5_re[child_ind] * ylm[5*ncomp + ychild_ind] + \
                        cd6_re[child_ind] * ylm[6*ncomp + ychild_ind] + \
                        cd7_re[child_ind] * ylm[7*ncomp + ychild_ind]) * coeff;
                    
                    const REAL child_im = (\
                        cd0_im[child_ind] * ylm[0*ncomp + ychild_ind_im] + \
                        cd1_im[child_ind] * ylm[1*ncomp + ychild_ind_im] + \
                        cd2_im[child_ind] * ylm[2*ncomp + ychild_ind_im] + \
                        cd3_im[child_ind] * ylm[3*ncomp + ychild_ind_im] + \
                        cd4_im[child_ind] * ylm[4*ncomp + ychild_ind_im] + \
                        cd5_im[child_ind] * ylm[5*ncomp + ychild_ind_im] + \
                        cd6_im[child_ind] * ylm[6*ncomp + ychild_ind_im] + \
                        cd7_im[child_ind] * ylm[7*ncomp + ychild_ind_im]) * coeff;


                    REAL re_mom, im_mom;
                    const INT64 ip = abs(kx) - abs(mx) - abs(kx - mx);
                    cplx_mul(child_re, child_im, IPOW_RE(ip), IPOW_IM(ip), &re_mom, &im_mom);
                    
                    //printf("n=%d; m=%d | A_n^m = %f\n", jx - nx, kx - mx, alm[(jx - nx)*4*nlevel + abs(kx - mx)]);

                    pd_re[CUBE_IND(jx, kx)] += re_mom;
                    pd_im[CUBE_IND(jx, kx)] += im_mom;

                }}
        }}




    }
/*
    printf("----------------\n");
    const INT64 ccc = 0;
    for(INT64 lx=0 ; lx<nlevel ; lx++){
        for(INT64 mx=-1*lx ; mx<=lx ; mx++){
            printf("l %d m %d ylm %f\n", lx, mx, ylm[ccc*ncomp + CUBE_IND(lx, mx)]);
        }
    }
*/



    return err;
}





