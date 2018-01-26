


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


static inline void rotate_p_moments(
    const INT32 p,
    const REAL * RESTRICT re_m,
    const REAL * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
){

    // implement complex matvec
    for(INT32 rx=0; rx<p ; rx++){
        REAL re_c = 0.0;
        REAL im_c = 0.0;
        for(INT32 cx=0; cx<p ; cx++){
            cplx_mul_add(   re_m[p*rx+cx],  im_m[p*rx+cx],
                            re_x[cx],       im_x[cx],
                            &re_c,          &im_c);
        }
        re_b[rx] = re_c;
        im_b[rx] = im_c;
    }
    return;
}

static inline void rotate_p_moments_append(
    const INT32 p,
    const REAL * RESTRICT re_m,
    const REAL * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
){

    // implement complex matvec
    for(INT32 rx=0; rx<p ; rx++){
        REAL re_c = 0.0;
        REAL im_c = 0.0;
        for(INT32 cx=0; cx<p ; cx++){
            cplx_mul_add(   re_m[p*rx+cx],  im_m[p*rx+cx],
                            re_x[cx],       im_x[cx],
                            &re_c,          &im_c);
        }
        re_b[rx] += re_c;
        im_b[rx] += im_c;
    }
    return;
}


// test wrapper for pth moment rotation
extern "C"
int rotate_p_moments_wrapper(
    const INT32 p,
    const REAL * RESTRICT re_m,
    const REAL * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
){
    rotate_p_moments( p, re_m, im_m, re_x, im_x, re_b, im_b);
    return 0;
}


static inline void rotate_moments(
    const INT32 p,
    const REAL * RESTRICT const * RESTRICT re_m,
    const REAL * RESTRICT const * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
){
    const INT32 im_offset = p*p;
    for(INT32 px=0 ; px<p ; px++){
        rotate_p_moments(2*px+1, re_m[px], im_m[px],
        &re_x[px*px], &im_x[px*px],
        &re_b[px*px], &im_b[px*px]);
    }
}

// test wrapper for rotate_moments
extern "C"
int rotate_moments_wrapper(
    const INT32 p,
    const REAL * RESTRICT const * RESTRICT re_m,
    const REAL * RESTRICT const * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
)
{
    rotate_moments(p, re_m, im_m, re_x, im_x, re_b, im_b);
    return 0;
}


static inline void rotate_moments_append(
    const INT32 p,
    const REAL * RESTRICT const * RESTRICT re_m,
    const REAL * RESTRICT const * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
){
    const INT32 im_offset = p*p;
    for(INT32 px=0 ; px<p ; px++){
        rotate_p_moments_append(2*px+1, re_m[px], im_m[px],
        &re_x[px*px], &im_x[px*px],
        &re_b[px*px], &im_b[px*px]);
    }
}



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



static inline void mtl_z(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT const * RESTRICT re_mat_forw,
    const REAL * RESTRICT const * RESTRICT im_mat_forw,
    const REAL * RESTRICT const * RESTRICT re_mat_back,
    const REAL * RESTRICT const * RESTRICT im_mat_back,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    const REAL * RESTRICT   i_array,
    REAL * RESTRICT         ldata,
    REAL * RESTRICT thread_space
){
    //const INT64 ASTRIDE1 = 4*nlevel + 1;
    //const INT64 ASTRIDE2 = 2*nlevel;

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 nlevel4 = nlevel*4;
    const INT64 im_offset = nlevel*nlevel;
    
    const INT64 nblk = 2*nlevel+2;
    REAL iradius_n[nblk];
    

    const REAL iradius = 1./radius;
    iradius_n[0] = 1.0;
    for(INT64 nx=1 ; nx<nblk ; nx++){ iradius_n[nx] = iradius_n[nx-1] * iradius; }

    REAL * RESTRICT iradius_p1 = &iradius_n[1];
    
    const INT32 ts = nlevel*nlevel;
    REAL * RESTRICT tmp_rel = &thread_space[0];
    REAL * RESTRICT tmp_iml = &thread_space[ts];
    REAL * RESTRICT tmp_reh = &thread_space[2*ts];
    REAL * RESTRICT tmp_imh = &thread_space[3*ts];
    
    
    // rotate foward
    rotate_moments(
        nlevel,
        re_mat_forw,
        im_mat_forw,
        odata,
        &odata[im_offset],
        tmp_rel,
        tmp_iml
    );
    

    for(INT32 jx=0 ; jx<ts ; jx++){
        tmp_reh[jx]=0.0;
        tmp_imh[jx]=0.0;
    }
    

    // loop over parent moments
    for(INT32 jx=0     ; jx<nlevel ; jx++ ){
    
        REAL * RESTRICT new_re = &tmp_reh[CUBE_IND(jx, 0)];
        REAL * RESTRICT new_im = &tmp_imh[CUBE_IND(jx, 0)];

        for(INT32 nx=0     ; nx<nlevel ; nx++){
        
            const INT32 kmax = MIN(nx, jx);
            const REAL ia_jn = ar_array[nx+jx];
            const REAL m1tn = IARRAY[nx];   // -1^{n}
            const REAL rr_jn1 = iradius_p1[jx+nx];     // 1 / rho^{j + n + 1}
            
            const REAL outer_coeff = ia_jn * m1tn * rr_jn1;

            for(INT32 kx=-1*kmax ; kx<=kmax    ; kx++){

                const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];     // A_j^k

                const REAL anm = a_array[nx*ASTRIDE1 + ASTRIDE2 + kx];
                
                const REAL ipower = IARRAY[kx];

                const REAL coeff_re = ipower * anm * ajk * outer_coeff;
                
                const INT64 oind = CUBE_IND(nx, kx);

                new_re[kx] += tmp_rel[oind] * coeff_re;
                new_im[kx] += tmp_iml[oind] * coeff_re;

            }

        }
    }


    rotate_moments_append(
        nlevel,
        re_mat_back,
        im_mat_back,
        tmp_reh,
        tmp_imh,
        ldata,
        &ldata[im_offset]
    );


}

extern "C"
int mtl_z_wrapper(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT const * RESTRICT re_mat_forw,
    const REAL * RESTRICT const * RESTRICT im_mat_forw,
    const REAL * RESTRICT const * RESTRICT re_mat_back,
    const REAL * RESTRICT const * RESTRICT im_mat_back,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   ar_array,
    const REAL * RESTRICT   i_array,
    REAL * RESTRICT         ldata,
    REAL * RESTRICT thread_space
){
    mtl_z(
        nlevel,
        radius,
        odata,
        re_mat_forw,
        im_mat_forw,
        re_mat_back,
        im_mat_back,
        a_array,
        ar_array,
        i_array,
        ldata,
        thread_space
    );

    return 0;
}


extern "C"
int translate_mtl(
    const UINT32 * RESTRICT dim_child,      // slowest to fastest
    const REAL * RESTRICT multipole_moments,
    REAL * RESTRICT local_moments,
    const REAL * RESTRICT const * RESTRICT const * RESTRICT re_mat_forw,
    const REAL * RESTRICT const * RESTRICT const * RESTRICT im_mat_forw,
    const REAL * RESTRICT const * RESTRICT const * RESTRICT re_mat_back,
    const REAL * RESTRICT const * RESTRICT const * RESTRICT im_mat_back,
    const REAL * RESTRICT alm,
    const REAL * RESTRICT almr,
    const REAL * RESTRICT i_array,
    const REAL radius,
    const INT64 nlevel,
    const INT32 * RESTRICT int_list,
    const INT32 * RESTRICT int_tlookup,
    const INT32 * RESTRICT int_plookup,
    const double * RESTRICT int_radius,
    REAL * RESTRICT * RESTRICT gthread_space
    ){
    int err = 0;
    const INT64 ncells = dim_child[0] * dim_child[1] * dim_child[2];

    const INT64 ncomp = nlevel*nlevel*2;
    const INT64 ncomp2 = nlevel*nlevel*8;
    const INT64 im_offset = nlevel*nlevel;
    const INT64 im_offset2 = 4*nlevel*nlevel;
    const UINT32 dim_halo[3] = {dim_child[0] + 4,
        dim_child[1] + 4, dim_child[2] + 4};
    const UINT32 dim_eight[3] = {2, 2, 2};

    const INT32 phi_stride = 8*nlevel + 2;
    const INT32 theta_stride = 4 * nlevel * nlevel;


    #pragma omp parallel for default(none) schedule(dynamic) \
    shared(dim_child, multipole_moments, local_moments, \
    alm, almr, i_array, int_list, int_tlookup, \
    int_plookup, int_radius, dim_eight, dim_halo, \
    re_mat_back, im_mat_back, im_mat_forw, re_mat_forw, gthread_space)
    for( INT64 pcx=0 ; pcx<ncells ; pcx++ ){
        INT64 cx, cy, cz;
        lin_to_xyz(dim_child, pcx, &cx, &cy, &cz);

        const int tid = omp_get_thread_num();
        REAL * RESTRICT thread_space = gthread_space[tid];

        // multipole moments are in a halo type
        const INT64 ccx = cx + 2;
        const INT64 ccy = cy + 2;
        const INT64 ccz = cz + 2;

        const INT64 halo_ind = xyz_to_lin(dim_halo, ccx, ccy, ccz);
        const INT64 octal_ind = xyz_to_lin(dim_eight, 
            cx & 1, cy & 1, cz & 1);

        
        REAL * out_moments = &local_moments[ncomp * pcx];
        // loop over contributing nearby cells.

        for( INT32 conx=octal_ind*189 ; conx<(octal_ind+1)*189 ; conx++ ){
            
            const REAL local_radius = int_radius[conx] * radius;
            const INT32 jcell = int_list[conx] + halo_ind;

            const INT32 t_lookup = int_tlookup[conx];

            mtl_z(nlevel, local_radius, &multipole_moments[jcell*ncomp],
                re_mat_forw[t_lookup],
                im_mat_forw[t_lookup],
                re_mat_back[t_lookup],
                im_mat_back[t_lookup],
                alm,
                almr,
                i_array,
                out_moments,
                thread_space);
           
        }
/*        
        printf("==========%d\t%d\t%d==========\n", cx, cy, cz);
        for(int jx=0; jx<nlevel ; jx++){
            for(int kx=-1*jx; kx<=jx ; kx++){
                printf("%d\t%d\t%f\n",jx, kx, out_moments[CUBE_IND(jx, kx)]);
            }
        }
*/        
    }

    return err;
}



