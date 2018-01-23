

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
    const REAL * RESTRICT * RESTRICT re_m,
    const REAL * RESTRICT * RESTRICT im_m,
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
    const REAL * RESTRICT * RESTRICT re_m,
    const REAL * RESTRICT * RESTRICT im_m,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_b,
    REAL * RESTRICT im_b
)
{
    rotate_moments(p, re_m, im_m, re_x, im_x, re_b, im_b);
    return 0;
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
    const REAL * RESTRICT   phi_data,
    const REAL * RESTRICT   theta_data,
    const REAL * RESTRICT   a_array,
    const REAL * RESTRICT   i_array,
    REAL * RESTRICT         ldata,
    const INT64 DEBUG0,
    const INT64 DEBUG1
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
                
                //if (DEBUG0 == 0){printf("");}

    for(INT32 kx=-1*jx ; kx<=jx    ; kx++){
        const REAL ajk = a_array[jx * ASTRIDE1 + ASTRIDE2 + kx];     // A_j^k
        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0;
        
        const bool pb = (DEBUG0 == 0 && jx == 0 && kx ==0 && DEBUG1 == 0);

        const INT32 abs_kx = ABS(kx);

        for(INT32 nx=0     ; nx<=abs_kx ; nx++){
                
            //if (pb){printf("nx=%d\n", nx);}

            const REAL m1tn = 1.0 - 2.0*((REAL)(nx & 1));   // -1^{n}
            const INT64 jxpnx = jx + nx;
            const INT64 p_ind_base = P_IND(jxpnx, 0);
            const REAL rr_jn1 = iradius_p1[jxpnx];     // 1 / rho^{j + n + 1}


            // compute translation coefficient
            // A_n^m

            const REAL anm = a_array[nx*ASTRIDE1 + ASTRIDE2 + kx];
            

            const REAL coeff_re = \
                i_array[(nlevel+kx)*(nlevel*2 + 1) + nlevel + kx] * \
                m1tn * anm * ajk * rr_jn1;
            
            const INT64 oind = CUBE_IND(nx, kx);
            const REAL ocoeff_re = odata[oind]              * coeff_re;
            const REAL ocoeff_im = odata[oind + im_offset]  * coeff_re;
            
            contrib_re += ocoeff_re;
            contrib_im += ocoeff_im;

        }
        
        ldata[CUBE_IND(jx, kx)] += contrib_re;
        ldata[CUBE_IND(jx, kx) + im_offset] += contrib_im;

    }}
}

extern "C"
int translate_mtl(
    const UINT32 * RESTRICT dim_child,      // slowest to fastest
    const REAL * RESTRICT multipole_moments,
    REAL * RESTRICT local_moments,
    const REAL * RESTRICT phi_data,
    const REAL * RESTRICT theta_data,
    const REAL * RESTRICT alm,
    const REAL * RESTRICT almr,
    const REAL * RESTRICT i_array,
    const REAL radius,
    const INT64 nlevel,
    const INT32 * RESTRICT int_list,
    const INT32 * RESTRICT int_tlookup,
    const INT32 * RESTRICT int_plookup,
    const double * RESTRICT int_radius,
    const INT32 * RESTRICT j_array,
    const INT32 * RESTRICT k_array,
    const REAL * RESTRICT a_inorder
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
    phi_data, theta_data, alm, almr, i_array, int_list, int_tlookup, \
    int_plookup, int_radius, dim_eight, dim_halo, j_array, k_array, a_inorder)
    for( INT64 pcx=0 ; pcx<ncells ; pcx++ ){
        INT64 cx, cy, cz;
        lin_to_xyz(dim_child, pcx, &cx, &cy, &cz);

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
            const INT32 p_lookup = int_plookup[conx];

            mtl_z(nlevel, local_radius, &multipole_moments[jcell*ncomp],
                &phi_data[p_lookup * phi_stride],
                &theta_data[t_lookup * theta_stride],
                alm, i_array,
                out_moments, pcx, conx);
           
        }
        
    }

    return err;
}



