

__constant__ INT32 d_nlevel;
__constant__ REAL d_radius;
__constant__ INT32 d_ncells;
__constant__ INT32 d_re_ncomp;

__constant__ UINT32 d_plain_dim0;
__constant__ UINT32 d_plain_dim1;
__constant__ UINT32 d_plain_dim2;

__constant__ INT32 d_phi_stride;
__constant__ INT32 d_theta_stride;

__constant__ INT32 d_ASTRIDE1;
__constant__ INT32 d_ASTRIDE2;


static inline __device__ void cplx_mul_add(
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

/*
static __global__ void mtl_kernel(
    const INT32 num_indices,
    const INT32 nblocks,
    const REAL * RESTRICT d_multipole_moments,
    const REAL * RESTRICT d_phi_data,
    const REAL * RESTRICT d_theta_data,
    const REAL * RESTRICT d_alm,
    const REAL * RESTRICT d_almr,
    const INT32 * RESTRICT d_int_list,
    const INT32 * RESTRICT d_int_tlookup,
    const INT32 * RESTRICT d_int_plookup,
    const double * RESTRICT d_int_radius,
    REAL * RESTRICT d_local_moments
){
    const INT32 plainx = blockIdx.x/nblocks;
    const INT32 plainy = blockIdx.y;
    const INT32 plainz = blockIdx.z;

    const INT32 local_base = 2*num_indices*(plainx + \
        d_plain_dim2*(plainy + d_plain_dim1*plainz));


    const INT32 index_id = (blockIdx.x % nblocks)*blockDim.x + threadIdx.x;
    const bool valid_id = (index_id < num_indices);
    const INT32 jx = valid_id ? dc_jlookup[index_id] : -1;
    const INT32 kx = valid_id ? dc_klookup[index_id] : -1;

    const INT32 halo_ind = (plainx + 2) + (d_plain_dim2+4)* \
        ( (plainy + 2) + (d_plain_dim1+4) * (plainz + 2) );

    const INT32 octal_ind = (plainx % 2) + \
        2*( (plainy % 2) + 2*(plainz % 2) );
    
    const INT32 jx_min = dc_jlookup[(blockIdx.x % nblocks)*blockDim.x];
    const INT32 jx_max = dc_jlookup[min(
        (blockIdx.x % nblocks + 1)*blockDim.x-1,
        num_indices-1
    )];

    REAL contrib_re = 0.0;
    REAL contrib_im = 0.0; 
    const REAL ajk =  valid_id ? d_alm[jx*d_ASTRIDE1 + d_ASTRIDE2 + kx] : 0.0;

    const INT32 absk = ABS(kx);

    // shared space for Y_{n+j}^{m-k}/A_{n+j}^{m-k}
    extern __shared__ REAL shared_y[];

    for (INT32 conx=octal_ind*189 ; conx<(octal_ind+1)*189 ; conx++){
            const REAL local_radius = d_int_radius[conx] * d_radius;
            
            const REAL iradius = 1./local_radius;

            const INT32 jcell = (d_int_list[conx] + halo_ind)*2*d_nlevel*d_nlevel;
            
            const REAL * RESTRICT d_phi_base = \
                &d_phi_data[d_phi_stride * d_int_plookup[conx]];
            const REAL * RESTRICT d_theta_base = \
                &d_theta_data[d_theta_stride * d_int_tlookup[conx]];
            
            //const bool pb = jx==0 && kx==0 && conx ==0 && plainx == 0 && plainy == 0 && plainz ==0;

            // we have blockDim.x threads
            for( INT32 jt=jx_min ; jt<jx_max+d_nlevel ; jt++ ){
                // need to loop over -jt to jt
                for( INT32 kt=-1*jt + threadIdx.x ; kt<=jt ; kt+=blockDim.x ) {
                    const INT32 out_ind = CUBE_IND(jt, kt);
                    const REAL ppart = d_theta_base[out_ind];
                    
                    shared_y[out_ind] = ppart * d_phi_base[EXP_RE_IND(2*d_nlevel, kt)];
                    shared_y[out_ind + 4*d_nlevel*d_nlevel] = ppart * d_phi_base[EXP_IM_IND(2*d_nlevel, kt)];
                    
                }
            }
            __syncthreads();
            
            REAL iradiuspj = pow(iradius, jx+1);
            REAL m1tn = 1.0;
            // use Y values
            for( INT32 nx=0 ; nx<d_nlevel ; nx++ ){
                
                //if (pb) { printf("nx=%d\n", nx);}

                for( INT32 mx=-1*nx ; mx<=nx ; mx++ ){
                    
                    //if (pb) { printf("\tmx=%d\n", mx); }

                    const REAL anm = d_alm[nx*d_ASTRIDE1 + d_ASTRIDE2 + mx];


                    const REAL ipkm = ((ABS(kx-mx) - absk - ABS(mx)) % 4) == 0 ? 1.0 : -1.0;
                    const REAL coeff = ipkm * anm * ajk * m1tn * iradiuspj;
                    const REAL o_re = coeff * d_multipole_moments[jcell + CUBE_IND(nx, mx)];
                    const REAL o_im = coeff * d_multipole_moments[jcell + CUBE_IND(nx, mx) + d_nlevel*d_nlevel];
                    
                    const INT32 yind = valid_id ? CUBE_IND(jx+nx, mx-kx) : 0;

                    cplx_mul_add(   o_re, o_im, 
                                    shared_y[yind], shared_y[yind+4*d_nlevel*d_nlevel],
                                    &contrib_re, &contrib_im);
                    
                    //if (pb) { printf("\t\tcuda=%d\t%f\n", jcell + CUBE_IND(nx, mx), iradiuspj);}
                }
                
                m1tn *= -1.0;
                iradiuspj *= iradius;
            }
            __syncthreads();
    }

    if(valid_id){
        //printf("local_base %d cube_ind %d jx %d kx %d\n", local_base, CUBE_IND(jx, kx), jx, kx);
        d_local_moments[local_base + CUBE_IND(jx, kx)] = contrib_re;
        d_local_moments[local_base + CUBE_IND(jx, kx) + d_nlevel*d_nlevel] = contrib_im;
    }
 
    //if (plainx == 0 && plainy == 0 && plainz == 0 && valid_id){
    //    printf("jx %d kx %d jx_min %d jx_max %d\n", jx, kx, jx_min, jx_max);
    //}

}
*/

static __global__ void mtl_kernel2(
    const INT32 num_indices,
    const INT32 nblocks,
    const REAL * RESTRICT d_multipole_moments,
    const REAL * RESTRICT d_phi_data,
    const REAL * RESTRICT d_theta_data,
    const REAL * RESTRICT d_alm,
    const REAL * RESTRICT d_almr,
    const INT32 * RESTRICT d_int_list,
    const INT32 * RESTRICT d_int_tlookup,
    const INT32 * RESTRICT d_int_plookup,
    const double * RESTRICT d_int_radius,
    const INT32 * RESTRICT d_jlookup,
    const INT32 * RESTRICT d_klookup,
    const REAL * RESTRICT d_ipower_mtl,
    REAL * RESTRICT d_local_moments
){
    const INT32 plainx = blockIdx.x/nblocks;
    const INT32 plainy = blockIdx.y;
    const INT32 plainz = blockIdx.z;



    const INT32 index_id = (blockIdx.x % nblocks)*blockDim.x + threadIdx.x;
    const bool valid_id = (index_id < num_indices);

    if (valid_id){
        const INT32 jx = d_jlookup[index_id];
        const INT32 kx = d_klookup[index_id];

        const INT32 octal_ind = (plainx % 2) + \
            2*( (plainy % 2) + 2*(plainz % 2) );

        REAL contrib_re = 0.0;
        REAL contrib_im = 0.0; 

        for (INT32 conx=octal_ind*189 ; conx<(octal_ind+1)*189 ; conx++){
            
            const REAL iradius = 1./(d_int_radius[conx] * d_radius);

            const INT32 jcell = (d_int_list[conx] + \
                ((plainx + 2) + (d_plain_dim2+4)* \
                ( (plainy + 2) + (d_plain_dim1+4) * (plainz + 2) )) \
                )*2*d_nlevel*d_nlevel;
            
            
            REAL m1tn_ajk = d_alm[jx*d_ASTRIDE1 + d_ASTRIDE2 + kx] * pow(iradius, jx+1);
            // use Y values
            for( INT32 nx=0 ; nx<d_nlevel ; nx++ ){

                for( INT32 mx=-1*nx ; mx<=nx ; mx++ ){
                    
                    // a_n_m
                    REAL coeff = d_alm[nx*d_ASTRIDE1 + d_ASTRIDE2 + mx] * \
                    // i*(k,m)
                    (((ABS(kx-mx) - ABS(kx) - ABS(mx)) % 4) == 0 ? 1.0 : -1.0) * \
                    // (-1)**(n) * A_j_k
                    m1tn_ajk;

                    const REAL o_re = coeff * d_multipole_moments[jcell + CUBE_IND(nx, mx)];
                    const REAL o_im = coeff * d_multipole_moments[jcell + CUBE_IND(nx, mx) + d_nlevel*d_nlevel];

                    const REAL ppart = d_theta_data[d_theta_stride * d_int_tlookup[conx] + \
                        CUBE_IND(jx+nx, mx-kx)];
                    
                    const REAL y_re = ppart * d_phi_data[d_phi_stride * d_int_plookup[conx] + \
                        EXP_RE_IND(2*d_nlevel, mx-kx)];

                    const REAL y_im = ppart * d_phi_data[d_phi_stride * d_int_plookup[conx] + \
                        EXP_IM_IND(2*d_nlevel, mx-kx)];


                    contrib_re += (o_re * y_re) - (o_im * y_im);
                    contrib_im += (y_re * o_im) + (o_re * y_im);
                    
                }

                
                m1tn_ajk *= -1.0 * iradius;
            }
        }

        const INT32 local_base = 2*num_indices*(plainx + \
        d_plain_dim2*(plainy + d_plain_dim1*plainz));   
        d_local_moments[local_base + CUBE_IND(jx, kx)] = contrib_re;
        d_local_moments[local_base + CUBE_IND(jx, kx) + d_nlevel*d_nlevel] = contrib_im;
    }


}




extern "C"
int translate_mtl(
    const UINT32 * RESTRICT dim_child,      // slowest to fastest
    const REAL * RESTRICT d_multipole_moments,
    REAL * RESTRICT d_local_moments,
    const REAL * RESTRICT d_phi_data,
    const REAL * RESTRICT d_theta_data,
    const REAL * RESTRICT d_alm,
    const REAL * RESTRICT d_almr,
    const REAL radius,
    const INT64 nlevel,
    const INT32 * RESTRICT d_int_list,
    const INT32 * RESTRICT d_int_tlookup,
    const INT32 * RESTRICT d_int_plookup,
    const double * RESTRICT d_int_radius,
    const INT32 * RESTRICT d_jlookup,
    const INT32 * RESTRICT d_klookup,
    const REAL * RESTRICT d_ipower_mtl,
    const INT32 thread_block_size
){


    int err = 0;
    //this is z, y, x
    const INT64 ncells = dim_child[0] * dim_child[1] * dim_child[2];

    //const INT64 ncomp = nlevel*nlevel*2;
    const INT32 re_ncomp = nlevel*nlevel;
    const INT64 ncomp2 = nlevel*nlevel*8;
    //const INT64 im_offset = nlevel*nlevel;
    //const INT64 im_offset2 = 4*nlevel*nlevel;


    const INT32 phi_stride = 8*nlevel + 2;
    const INT32 theta_stride = 4 * nlevel * nlevel;
    
    const INT32 ASTRIDE1 = 4*nlevel + 1;
    const INT32 ASTRIDE2 = 2*nlevel;

    
    const INT32 num_indices = nlevel * nlevel;

    const INT64 nblocks = ((num_indices%thread_block_size) == 0) ? \
        num_indices/thread_block_size : \
        num_indices/thread_block_size + 1;
    
    dim3 grid_block(nblocks*dim_child[2], dim_child[1], dim_child[0]);
    dim3 thread_block(thread_block_size, 1, 1);


    int device_id = -1;
    err = cudaGetDevice(&device_id);
    if (err != cudaSuccess){return err;}

    cudaDeviceProp device_prop;
    err = cudaGetDeviceProperties(&device_prop, device_id);
    if (err != cudaSuccess){return err;}

    if (device_prop.maxThreadsPerBlock<thread_block_size) { 
        printf("bad threadblock size: %d, device max: %d\n", thread_block_size, device_prop.maxThreadsPerBlock); 
        return cudaErrorUnknown; 
    }
 
    err = cudaMemcpyToSymbol(d_nlevel, &nlevel, sizeof(INT32));
    if (err != cudaSuccess) {return err;}
    
    err = cudaMemcpyToSymbol(d_radius, &radius, sizeof(REAL));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_ncells, &ncells, sizeof(INT32));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_re_ncomp, &re_ncomp, sizeof(INT32));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_plain_dim0, &dim_child[0], sizeof(UINT32));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpyToSymbol(d_plain_dim1, &dim_child[1], sizeof(UINT32));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpyToSymbol(d_plain_dim2, &dim_child[2], sizeof(UINT32));
    if (err != cudaSuccess) {return err;}
    
    err = cudaMemcpyToSymbol(d_phi_stride, &phi_stride, sizeof(INT32));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpyToSymbol(d_theta_stride, &theta_stride, sizeof(INT32));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_ASTRIDE1, &ASTRIDE1, sizeof(INT32));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_ASTRIDE2, &ASTRIDE2, sizeof(INT32));
    if (err != cudaSuccess) {return err;}


    const size_t shared_bytes = sizeof(REAL) * ncomp2;

    //cudaStream_t stream;
    //cudaStreamCreate (&stream);

    //mtl_kernel<<<grid_block, thread_block, shared_bytes>>>(
    //mtl_kernel2<<<grid_block, thread_block, 0, stream>>>(
    mtl_kernel2<<<grid_block, thread_block>>>(
        num_indices,
        nblocks,
        d_multipole_moments,
        d_phi_data,
        d_theta_data,
        d_alm,
        d_almr,
        d_int_list,
        d_int_tlookup,
        d_int_plookup,
        d_int_radius,
        d_jlookup,
        d_klookup,
        d_ipower_mtl,        
        d_local_moments
    );
    
    //err = cudaStreamSynchronize(stream);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return err;}

    return err;
}




