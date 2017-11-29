
__constant__ INT32 d_ncells;
__constant__ INT32 d_re_ncomp;
__constant__ INT32 * RESTRICT dc_jlookup;
__constant__ INT32 * RESTRICT dc_klookup;

__constant__ UINT32 d_plain_dim0;
__constant__ UINT32 d_plain_dim1;
__constant__ UINT32 d_plain_dim2;

static inline __device__ void lin_to_plain_xyz(
    const INT32 lin,
    INT32 * RESTRICT cx,
    INT32 * RESTRICT cy,
    INT32 * RESTRICT cz
){
    *cx = lin % d_plain_dim2;
    const INT32 yz = (lin - (*cx))/d_plain_dim2;
    *cy = yz % d_plain_dim1;
    *cz = (yz - (*cy))/d_plain_dim1;
}

static inline __device__ INT32 get_octal_lin(
    const INT32 cx,
    const INT32 cy,
    const INT32 cz
){
    const INT32 cxt = (cx + 2) % 2;
    const INT32 cyt = (cy + 2) % 2;
    const INT32 czt = (cz + 2) % 2;

    return cxt + 2*(cyt + 2*czt);
}


static inline __device__ INT32 xyz_to_halo_lin(
    const INT32 cx,
    const INT32 cy,
    const INT32 cz
){
    return cx + (d_plain_dim2+4)*(cy + (d_plain_dim1 + 4)*cz);
}


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

__device__ INT32 get_cell_id(const INT32 gid){
    return gid/d_re_ncomp;
}

__device__ INT32 get_local_cell_id(const INT32 gid){
    return gid % d_re_ncomp;
}

__device__ INT32 get_j(const INT32 local_cell_id){
    return dc_jlookup[local_cell_id];
}

__device__ INT32 get_k(const INT32 local_cell_id){
    return dc_klookup[local_cell_id];
}

static __global__ void mtl_kernel(
    const INT32 max_thread_id,
    const REAL * RESTRICT d_multipole_moments,
    const REAL * RESTRICT d_phi_data,
    const REAL * RESTRICT d_theta_data,
    const REAL * RESTRICT d_alm,
    const REAL * RESTRICT d_almr,
    const REAL radius,   
    const INT32 * RESTRICT d_int_list,
    const INT32 * RESTRICT d_int_tlookup,
    const INT32 * RESTRICT d_int_plookup,
    const double * RESTRICT d_int_radius,
    REAL * RESTRICT d_local_moments
){
    const INT32 gid = blockIdx.x *blockDim.x + threadIdx.x;
    if ( gid<max_thread_id ){
        const INT32 cell_id = get_cell_id(gid);
        const INT32 local_cell_id = get_local_cell_id(gid);
        const INT32 jx = get_j(local_cell_id);
        const INT32 kx = get_k(local_cell_id);
        
        INT32 plainx, plainy, plainz;
        lin_to_plain_xyz(cell_id, &plainx, &plainy, &plainz);
        const INT32 octal_ind = get_octal_lin(plainx, plainy, plainz);
        const INT32 halo_ind = xyz_to_halo_lin(plainx+2, plainy+2, plainz+2);


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
    const INT32 RESTRICT d_jlookup,
    const INT32 RESTRICT d_klookup,
    const INT32 thread_block_size
){


    int err = 0;
    const INT64 ncells = dim_child[0] * dim_child[1] * dim_child[2];

    const INT64 ncomp = nlevel*nlevel*2;
    const INT32 re_ncomp = nlevel*nlevel;
    const INT64 ncomp2 = nlevel*nlevel*8;
    const INT64 im_offset = nlevel*nlevel;
    const INT64 im_offset2 = 4*nlevel*nlevel;
    const UINT32 dim_halo[3] = {dim_child[0] + 4,
        dim_child[1] + 4, dim_child[2] + 4};
    const UINT32 dim_eight[3] = {2, 2, 2};

    const INT32 phi_stride = 8*nlevel + 2;
    const INT32 theta_stride = 4 * nlevel * nlevel;

    
    const INT32 num_indices = ncells * nlevel * nlevel;

    const INT64 nblocks = num_indices/thread_block_size + 1;
    

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

    if (device_prop.maxGridSize[0]<nblocks) { 
        printf("bad grid size: %d, device max: %d\n", nblocks, device_prop.maxGridSize[0]); 
        return cudaErrorUnknown; 
    }
    
    err = cudaMemcpyToSymbol(d_ncells, &ncells, sizeof(INT32));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_re_ncomp, &re_ncomp, sizeof(INT32));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_re_ncomp, &re_ncomp, sizeof(INT32));
    if (err != cudaSuccess) {return err;}
    
    err = cudaMemcpyToSymbol(dc_jlookup, &d_jlookup, sizeof(INT32*));
    if (err != cudaSuccess) {return err;}
    
    err = cudaMemcpyToSymbol(dc_klookup, &d_klookup, sizeof(INT32*));
    if (err != cudaSuccess) {return err;}

    err = cudaMemcpyToSymbol(d_plain_dim0, &dim_child[0], sizeof(UINT32));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpyToSymbol(d_plain_dim1, &dim_child[1], sizeof(UINT32));
    if (err != cudaSuccess) {return err;}
    err = cudaMemcpyToSymbol(d_plain_dim2, &dim_child[2], sizeof(UINT32));
    if (err != cudaSuccess) {return err;}


    mtl_kernel<<<nblocks, thread_block_size>>>(
        num_indices,
        d_multipole_moments,
        d_phi_data,
        d_theta_data,
        d_alm,
        d_almr,
        radius,   
        d_int_list,
        d_int_tlookup,
        d_int_plookup,
        d_int_radius,
        d_local_moments
    );
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return err;}

    return err;
}




