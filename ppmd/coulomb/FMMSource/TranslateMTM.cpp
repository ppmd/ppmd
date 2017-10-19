

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

static inline double J(
    const INT64 m, 
    const INT64 mp
){
    return (m*mp < 0) ? pow(-1.0, MIN(m, mp)) : 1.0;
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
    cout << radius_n[0] << endl;
    for(INT64 nx=1 ; nx<nlevel ; nx++){
        radius_n[nx] = radius_n[nx-1] * radius;
        cout << nx << " " << radius_n[nx] << endl;

    } 


    printf("Warning loop bounds overrode!\n");
    #pragma omp parallel for default(none) schedule(dynamic) shared(dim_parent,\
    dim_child,moments_child,moments_parent,ylm,alm,almr, radius_n, ipow_re, ipow_im)
    //for( INT64 pcx=0 ; pcx<nparent_cells ; pcx++ ){
    for( INT64 pcx=0 ; pcx<1 ; pcx++ ){
        INT64 cx, cy, cz;
        lin_to_xyz(dim_parent, pcx, &cx, &cy, &cz);
        
        REAL * RESTRICT pd_re = &moments_parent[pcx*ncomp];
        REAL * RESTRICT pd_im = &moments_parent[pcx*ncomp + im_offset];

        // child layer has halos
        const INT64 ccx = 2*cx + 2;
        const INT64 ccy = 2*cy + 2;
        const INT64 ccz = 2*cz + 2;

        //children are labeled lexicographically
        const INT64 cc0 = ncomp * xyz_to_lin(dim_child, ccx, ccy, ccz);
        //const INT64 cc1 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy, ccz);
        //const INT64 cc2 = ncomp * xyz_to_lin(dim_child, ccx, ccy+1, ccz);
        //const INT64 cc3 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy+1, ccz);
        //const INT64 cc4 = ncomp * xyz_to_lin(dim_child, ccx, ccy, ccz+1);
        //const INT64 cc5 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy, ccz+1);
        //const INT64 cc6 = ncomp * xyz_to_lin(dim_child, ccx, ccy+1, ccz+1);
        //const INT64 cc7 = ncomp * xyz_to_lin(dim_child, ccx+1, ccy+1, ccz+1);
        
        const REAL * RESTRICT cd0_re = &moments_child[cc0];
        //const REAL * RESTRICT cd1_re = &moments_child[cc1];
        //const REAL * RESTRICT cd2_re = &moments_child[cc2];
        //const REAL * RESTRICT cd3_re = &moments_child[cc3];
        //const REAL * RESTRICT cd4_re = &moments_child[cc4];
        //const REAL * RESTRICT cd5_re = &moments_child[cc5];
        //const REAL * RESTRICT cd6_re = &moments_child[cc6];
        //const REAL * RESTRICT cd7_re = &moments_child[cc7];
        
        // loop over parent moments
        for(INT64 jx=0     ; jx<nlevel ; jx++ ){
        for(INT64 kx=-1*jx ; kx<=jx    ; kx++){
                const REAL ajkr = almr[jx*4*nlevel + abs(kx)];
                for(INT64 nx=0     ; nx<=jx ; nx++){
                for(INT64 mx=-1*nx ; mx<=nx ; mx++){
                    
                    const bool km_jn = abs(kx - mx) <= (jx-nx);
                    const REAL a_jnkm = km_jn ? alm[(jx - nx)*4*nlevel + abs(kx - mx)] : 0.0;

                    const REAL coeff = ajkr * radius_n[nx] * alm[nx*4*nlevel + abs(mx)] * a_jnkm;

                    const INT64 child_ind = CUBE_IND(jx - nx, kx - mx);
                    const INT64 child_ind_im = child_ind + im_offset;

                    const INT64 ychild_ind = CUBE_IND(nx, -1*mx);
                    const INT64 ychild_ind_im = ychild_ind + im_offset;

                    REAL child_re = 0.0;
                    REAL child_im = 0.0;
                    
                    REAL mul_re;
                    REAL mul_im;
                    
                    REAL child_val_re;
                    REAL child_val_im;
                    child_val_re = km_jn ? cd0_re[child_ind] : 0.0;
                    child_val_im = km_jn ? cd0_re[child_ind_im] : 0.0;

                    cplx_mul(child_val_re, child_val_im, ylm[0*ncomp + ychild_ind], ylm[0*ncomp + ychild_ind_im], &mul_re, &mul_im);



                    child_re+=mul_re;
                    child_im+=mul_im;

                
                    child_re*=coeff;
                    child_im*=coeff;

                    REAL re_mom, im_mom;
                    const INT64 ip = abs(kx) - abs(mx) - abs(kx - mx);
                    const REAL icoeff = (ip < 0) ? -1.0 : 1.0;
                    const REAL jre = IPOW_RE(ip);
                    const REAL jim = IPOW_IM(ip)*icoeff;
                    
                    //const REAL jre = J(mx, kx-mx);
                    //const REAL jim = jre;

                    cplx_mul(child_re, child_im, jre, jim, &re_mom, &im_mom);
                    if (jx<2){
                        printf("j = %d\tk = %d\tn = %d\tm = %d\t| j-n=%d\tk-m=%d\tO= %f\t| val = %f \n", jx, kx, nx, mx, jx-nx, kx-mx, cd0_re[child_ind], re_mom);
                    }                    

                    pd_re[CUBE_IND(jx, kx)] += re_mom;
                    pd_im[CUBE_IND(jx, kx)] += im_mom;

                }}
        }}
/*
        for(INT64 jx=0     ; jx<nlevel ; jx++ ){
        for(INT64 kx=-1*jx ; kx<=jx    ; kx++){
                    pd_re[CUBE_IND(jx, kx)] = cd0_re[CUBE_IND(jx, kx)];
                    pd_im[CUBE_IND(jx, kx)] = cd0_re[CUBE_IND(jx, kx) + im_offset];

        }}
*/
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





