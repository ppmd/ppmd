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

static inline void rotate_p_forward(
    const INT32 p,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
){
    // rotate negative terms around z axis
    for(INT32 rx=0 ; rx<p ; rx++){
         cplx_mul(
            re_x[rx], im_x[rx],
            exp_re[p-1-rx],
            -1.0*exp_im[p-1-rx],
            &re_bz[rx], &im_bz[rx]
        );
    }
    re_bz[p] = re_x[p];
    im_bz[p] = im_x[p];
    // rotate positive terms around z axis
    for(INT32 rx=0 ; rx<p ; rx++){
         cplx_mul(
            re_x[p+1+rx], im_x[p+1+rx],
            exp_re[rx],
            exp_im[rx],
            &re_bz[p+1+rx], &im_bz[p+1+rx]
        );
    }
    //rotate around y axis
    // b <- (Wigner_d) * x
    const INT32 n = 2*p+1;
    for(INT32 rx=0 ; rx<p ; rx++){
        REAL hre = 0.0;
        REAL him = 0.0;
        REAL lre = 0.0;
        REAL lim = 0.0;
        for(INT32 cx=0 ; cx<n ; cx++){
            const REAL a = wig_forw[rx*n + cx];
            hre += a * re_bz[cx];
            him += a * im_bz[cx];
            lre += a * re_bz[n-cx-1];
            lim += a * im_bz[n-cx-1];
        }
        re_by[rx] = hre;
        im_by[rx] = him;
        re_by[n-rx-1] = lre;
        im_by[n-rx-1] = lim;
    }
    // middle row
    REAL mre = 0.0;
    REAL mim = 0.0;
    for( INT32 cx=0 ; cx<n ; cx++ ){
        const REAL a = wig_forw[p*n + cx];
        mre += a * re_bz[cx];
        mim += a * im_bz[cx];
    }
    re_by[p] = mre;
    im_by[p] = mim;
}

static inline void rotate_moments_forward(
    const INT32 l,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
){
    const INT32 im_offset = l*l;
    for(INT32 lx=0 ; lx<l ; lx++){
         rotate_p_forward(lx, exp_re, exp_im, wig_forw[lx],
         &re_x[lx*lx], &im_x[lx*lx],
         &re_bz[lx*lx], &im_bz[lx*lx],
         &re_by[lx*lx], &im_by[lx*lx]);       
    }
}

extern "C"
int wrapper_rotate_p_forward(
    const INT32 p,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
){
    rotate_p_forward(p, exp_re, exp_im, wig_forw, re_x,
        im_x, re_bz, im_bz, re_by, im_by);
    return 0;
}

// test wrapper for rotate_moments_forward
extern "C"
int wrapper_rotate_moments_forward(
    const INT32 l,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
)
{
    rotate_moments_forward(l, exp_re, exp_im, wig_forw, re_x,
        im_x, re_bz, im_bz, re_by, im_by);
    return 0;
}











static inline void rotate_p_backward(
    const INT32 p,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz
){
    //rotate around y axis
    // b <- (Wigner_d) * x
    const INT32 n = 2*p+1;
    for(INT32 rx=0 ; rx<p ; rx++){
        REAL hre = 0.0;
        REAL him = 0.0;
        REAL lre = 0.0;
        REAL lim = 0.0;
        for(INT32 cx=0 ; cx<n ; cx++){
            const REAL a = wig_forw[rx*n + cx];
            hre += a * re_x[cx];
            him += a * im_x[cx];
            lre += a * re_x[n-cx-1];
            lim += a * im_x[n-cx-1];
        }
        re_by[rx] = hre;
        im_by[rx] = him;
        re_by[n-rx-1] = lre;
        im_by[n-rx-1] = lim;
    }
    // middle row
    REAL mre = 0.0;
    REAL mim = 0.0;
    for( INT32 cx=0 ; cx<n ; cx++ ){
        const REAL a = wig_forw[p*n + cx];
        mre += a * re_x[cx];
        mim += a * im_x[cx];
    }
    re_by[p] = mre;
    im_by[p] = mim;
    
    // rotate negative terms around z axis
    for(INT32 rx=0 ; rx<p ; rx++){
         cplx_mul_add(
            re_by[rx], im_by[rx],
            exp_re[p-1-rx], exp_im[p-1-rx],
            &re_bz[rx], &im_bz[rx]
        );
    }
    re_bz[p] += re_by[p];
    im_bz[p] += im_by[p];
    // rotate positive terms around z axis
    for(INT32 rx=0 ; rx<p ; rx++){
         cplx_mul_add(
            re_by[p+1+rx], im_by[p+1+rx],
            exp_re[rx], -1.0*exp_im[rx],
            &re_bz[p+1+rx], &im_bz[p+1+rx]
        );
    }

}

static inline void rotate_moments_backward(
    const INT32 l,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz
){
    const INT32 im_offset = l*l;
    for(INT32 lx=0 ; lx<l ; lx++){
         rotate_p_backward(
         lx, exp_re, exp_im, wig_forw[lx],
         &re_x[lx*lx], &im_x[lx*lx],
         &re_by[lx*lx], &im_by[lx*lx],
         &re_bz[lx*lx], &im_bz[lx*lx]
        );       
    }
}

// test wrapper for rotate_moments_forward
extern "C"
int wrapper_rotate_moments_backward(
    const INT32 l,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT re_x,
    const REAL * RESTRICT im_x,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by,
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz
){
    rotate_moments_backward(l, exp_re, exp_im, wig_forw, re_x,
        im_x, re_by, im_by, re_bz, im_bz);
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
    const REAL * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT const * RESTRICT wig_back,
    const REAL * RESTRICT   exp_re,
    const REAL * RESTRICT   exp_im,
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
    rotate_moments_forward(nlevel,
        exp_re, exp_im, wig_forw,
        odata, &odata[im_offset],
        tmp_reh, tmp_imh,
        tmp_rel, tmp_iml
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
    
    // rotate backwards
    rotate_moments_backward(nlevel, 
        exp_re, exp_im, wig_back,
        tmp_reh, tmp_imh,
        tmp_rel, tmp_iml,
        ldata, &ldata[im_offset]
    );

}







extern "C"
int mtl_z_wrapper(
    const INT64             nlevel,
    const REAL              radius,
    const REAL * RESTRICT   odata,
    const REAL * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT const * RESTRICT wig_back,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
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
        wig_forw,
        wig_back,
        exp_re,
        exp_im,
        a_array,
        ar_array,
        i_array,
        ldata,
        thread_space
    );

    return 0;
}





static inline void blocked_forw_matvec(
    const INT64 block_size,
    const INT64 im_offset,
    const INT64 stride,
    const INT32 p,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT wig_forw,
    REAL const * RESTRICT re_x[BLOCK_SIZE],
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
){
    for( INT64 blk=0 ; blk<block_size ; blk++){
        // rotate negative terms around z axis
        for(INT32 rx=0 ; rx<p ; rx++){
             cplx_mul(
                re_x[blk][rx], re_x[blk][rx+im_offset],
                exp_re[p-1-rx], -1.0*exp_im[p-1-rx],
                &re_bz[rx+blk*stride], &im_bz[rx+blk*stride]
            );
        }
        re_bz[p+blk*stride] = re_x[blk][p];
        im_bz[p+blk*stride] = re_x[blk][p+im_offset];
        // rotate positive terms around z axis
        for(INT32 rx=0 ; rx<p ; rx++){
             cplx_mul(
                re_x[blk][p+1+rx], re_x[blk][p+1+rx+im_offset],
                exp_re[rx], exp_im[rx],
                &re_bz[p+1+rx+blk*stride], &im_bz[p+1+rx+blk*stride]
            );
        }
    }
    // naive matmul
    const INT32 n = 2*p+1;
    for( INT64 blk=0 ; blk<block_size ; blk++){
        for(INT32 rx=0 ; rx<p ; rx++){
            REAL hre = 0.0;
            REAL him = 0.0;
            REAL lre = 0.0;
            REAL lim = 0.0;
            for(INT32 cx=0 ; cx<n ; cx++){
                const REAL a = wig_forw[rx*n + cx];
                hre += a * re_bz[cx+blk*stride];
                him += a * im_bz[cx+blk*stride];
                lre += a * re_bz[n-cx-1+blk*stride];
                lim += a * im_bz[n-cx-1+blk*stride];
            }
            re_by[rx+blk*stride] = hre;
            im_by[rx+blk*stride] = him;
            re_by[n-rx-1+blk*stride] = lre;
            im_by[n-rx-1+blk*stride] = lim;
        }
        // middle row
        REAL mre = 0.0;
        REAL mim = 0.0;
        for( INT32 cx=0 ; cx<n ; cx++ ){
            const REAL a = wig_forw[p*n + cx];
            mre += a * re_bz[cx+blk*stride];
            mim += a * im_bz[cx+blk*stride];
        }
        re_by[p+blk*stride] = mre;
        im_by[p+blk*stride] = mim;
    }
}

extern "C"
int wrapper_blocked_forw_matvec(
    const INT64 block_size,
    const INT64 im_offset,
    const INT64 stride,
    const INT32 p,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT wig_forw,
    REAL const * RESTRICT re_x[BLOCK_SIZE],
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
){
    blocked_forw_matvec( block_size, im_offset, stride, p, exp_re,
            exp_im, wig_forw, re_x, re_bz, im_bz, re_by, im_by
    );
    return 0;
}

static inline void blocked_rotate_forward(
    const INT64 block_size,
    const INT64 l,
    const REAL * RESTRICT exp_re,
    const REAL * RESTRICT exp_im,
    const REAL * RESTRICT const * RESTRICT wig_forw,
    REAL const * RESTRICT in_ptrs[BLOCK_SIZE],
    REAL * RESTRICT re_bz,
    REAL * RESTRICT im_bz,
    REAL * RESTRICT re_by,
    REAL * RESTRICT im_by
){
    const INT64 im_offset = l*l;
    const INT64 stride = 2*im_offset;
    for(INT64 lx=0 ; lx<l ; lx++){
        blocked_forw_matvec(
            block_size,
            im_offset,
            stride,
            lx,
            exp_re,
            exp_im,
            wig_forw[lx],
            in_ptrs,
            &re_bz[lx*lx], &im_bz[lx*lx],
            &re_by[lx*lx], &im_by[lx*lx]
        );

    }
}


extern "C"
int translate_mtl(
    const UINT32 * RESTRICT dim_child,      // slowest to fastest
    const REAL * RESTRICT multipole_moments,
    REAL * RESTRICT local_moments,
    const REAL * RESTRICT const * RESTRICT const * RESTRICT wig_forw,
    const REAL * RESTRICT const * RESTRICT const * RESTRICT wig_back,
    const REAL * RESTRICT const * RESTRICT exp_re,
    const REAL * RESTRICT const * RESTRICT exp_im,
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
    

    const INT64 block_size = BLOCK_SIZE;
    const INT64 block_count = ncells/block_size;
    const INT64 block_end = block_count*block_size;
    
    #pragma omp parallel for default(none) schedule(dynamic) \
    shared(dim_child, multipole_moments, local_moments, \
    alm, almr, i_array, int_list, int_tlookup, \
    int_plookup, int_radius, dim_eight, dim_halo, \
    wig_forw, wig_back, exp_re, exp_im, gthread_space)
    for( INT64 blk=0 ; blk<block_end ; blk+=block_size ){

        const int tid = omp_get_thread_num();
        REAL * RESTRICT thread_space = gthread_space[tid];

        REAL * RESTRICT blk_out = thread_space;
        REAL * RESTRICT blk_tmp_start = blk_out + ncomp*block_size;
        
        // zero output moments
        for( INT64 ncx=0 ; ncx<ncomp*block_size; ncx++ ){
            blk_out[ncx] = 0.0;
        }
        // moments to translate
        REAL const * RESTRICT in_ptrs[BLOCK_SIZE];
        

        for( INT32 conx=0 ; conx<98 ; conx++ ){

            const REAL local_radius = int_radius[conx] * radius;
            const INT32 t_lookup = int_tlookup[conx];
            
            INT64 tblk = 0;
            for( INT64 pcx=blk ; pcx<(blk+block_size) ; pcx++ ){
                INT64 cx, cy, cz;
                lin_to_xyz(dim_child, pcx, &cx, &cy, &cz);
                // multipole moments are in a halo type
                const INT64 ccx = cx + 2;
                const INT64 ccy = cy + 2;
                const INT64 ccz = cz + 2;
                const INT64 halo_ind = xyz_to_lin(dim_halo, ccx, ccy, ccz);
                const INT32 jcell = int_list[conx] + halo_ind;
                in_ptrs[tblk] = &multipole_moments[jcell*ncomp];
                tblk++;
            }

        }

        // append new moments to output cell moments
        INT64 tblk = 0;
        for( INT64 pcx=blk ; pcx<(blk+block_size) ; pcx++ ){
            REAL * out_moments = &local_moments[ncomp * pcx];
            for( INT64 ncx=0 ; ncx<ncomp ; ncx++){
                out_moments[ncx] += blk_out[tblk*ncomp + ncx];
            }
            tblk++;
        }

    }

    // peel loop
    #pragma omp parallel for default(none) schedule(dynamic) \
    shared(dim_child, multipole_moments, local_moments, \
    alm, almr, i_array, int_list, int_tlookup, \
    int_plookup, int_radius, dim_eight, dim_halo, \
    wig_forw, wig_back, exp_re, exp_im, gthread_space)
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
        
        INT64 cstart = octal_ind*189;
        if (pcx < block_end) { cstart += 98; }
        
        REAL * out_moments = &local_moments[ncomp * pcx];
        // loop over contributing nearby cells.
		
        for( INT32 conx=cstart ; conx<(octal_ind+1)*189 ; conx++ ){
            
            const REAL local_radius = int_radius[conx] * radius;
            const INT32 jcell = int_list[conx] + halo_ind;

            const INT32 t_lookup = int_tlookup[conx];

            mtl_z(nlevel, local_radius, &multipole_moments[jcell*ncomp],
                wig_forw[t_lookup],
                wig_back[t_lookup],
                exp_re[t_lookup],
                exp_im[t_lookup],
                alm,
                almr,
                i_array,
                out_moments,
                thread_space);
           
        }

    }

    return err;
}



