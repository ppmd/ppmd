

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
    const INT32 thread_block_size,
    const INT32 device_number
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
    
    //err = cudaSetDevice(device_number);
    if (err != cudaSuccess){return err;}

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
    

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return err;}

    return err;
}




