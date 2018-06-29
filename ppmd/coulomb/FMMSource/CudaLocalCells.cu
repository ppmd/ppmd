


__global__ void zero_outputs(
    const INT64 nlocal,
    REAL * RESTRICT d_forces,
    REAL * RESTRICT d_potential_array
){  
    const INT64 threadid = threadIdx.x + blockIdx.x*blockDim.x;
    if (threadid<nlocal){
        d_potential_array[threadid] = 0.0;
        d_forces[threadid] = 0.0;
        d_forces[threadid + nlocal] = 0.0;
        d_forces[threadid + 2*nlocal] = 0.0;
    }

    return;
}








extern "C"
int local_cell_by_cell_0(
    const INT64 free_space,
    const REAL * RESTRICT extent,
    const INT64 * RESTRICT global_size,
    const INT64 * RESTRICT local_size,
    const INT64 * RESTRICT local_offset,
    const INT64 num_threads,
    const INT64 nlocal,
    const INT64 ntotal,
    const REAL * RESTRICT P,
    const REAL * RESTRICT Q,
    const INT64 * RESTRICT C,
    INT64 * RESTRICT ll_array,
    INT64 * RESTRICT ll_ccc_array,
    const INT64 thread_block_size,
    const INT64 device_number,
    REAL * RESTRICT d_positions,
    REAL * RESTRICT d_charges,
    REAL * RESTRICT d_forces,
    REAL * RESTRICT d_potential_array,
    INT64 * RESTRICT req_len,
    INT64 * RESTRICT ret_max_cell_count
){
    
    int err = 0;
    dim3 thread_block(thread_block_size, 1, 1);

    const INT64 padg = 3;
    const INT64 padl = 1;

    const INT64 plsx = local_size[2] + 2*padl;
    const INT64 plsy = local_size[1] + 2*padl;
    const INT64 plsz = local_size[0] + 2*padl;

    const INT64 pgsx = global_size[2] + 2*padg;
    const INT64 pgsy = global_size[1] + 2*padg;
    const INT64 pgsz = global_size[0] + 2*padg;

    const INT64 ncells_local = plsx*plsy*plsz;

    const INT64 ncells_global = global_size[0]*global_size[1]*global_size[2];
    const INT64 ncells_padded = pgsx * pgsy * pgsz;
    INT64 max_cell_count = 0;
    

    const INT64 nblocks = ((nlocal%thread_block_size) == 0) ? \
        nlocal/thread_block_size : \
        nlocal/thread_block_size + 1;

    dim3 grid_block(nblocks, 1, 1);
    
    err = cudaSetDevice(device_number);
    CUDACHECKERR

    int device_id = -1;
    err = cudaGetDevice(&device_id);
    CUDACHECKERR

    cudaDeviceProp device_prop;
    err = cudaGetDeviceProperties(&device_prop, device_id);
    CUDACHECKERR

    if (device_prop.maxThreadsPerBlock<thread_block_size) { 
        printf("bad threadblock size: %d, device max: %d\n", thread_block_size, device_prop.maxThreadsPerBlock); 
        return cudaErrorUnknown; 
    }   
    
    cudaStream_t s1, s2, s3;
    err = cudaStreamCreate(&s1);
    CUDACHECKERR
    err = cudaStreamCreate(&s2);
    CUDACHECKERR    
    err = cudaStreamCreate(&s3);
    CUDACHECKERR

    err = cudaMemcpyAsync((void*) d_positions, (void*) P, (size_t) 3*ntotal*sizeof(REAL),
            cudaMemcpyHostToDevice, s1); 
    CUDACHECKERR

    err = cudaMemcpyAsync (d_charges, Q, ntotal*sizeof(REAL), cudaMemcpyHostToDevice, s2); 
    CUDACHECKERR

    zero_outputs<<<grid_block, thread_block, 0, s3>>>(nlocal, d_forces, d_potential_array);

    // bin particles into fmm cells whilst copying is happening.


    omp_set_num_threads(num_threads);
    REAL energy = 0.0;
    INT64 part_count = 0;
    
    // pad global and pad local


    // padded by one cell to include particles that are allowed to drift out
    // of this domain due to cell list rebuilding.

    const INT64 ll_cend   = ncells_padded + ntotal;
    const INT64 ll_cstart = ntotal;


    const INT64 shift_x = 1;
    const INT64 shift_y = pgsx;
    const INT64 shift_z = pgsx*pgsy;
    
    const INT64 hshift_x = global_size[2];
    const INT64 hshift_y = global_size[1];
    const INT64 hshift_z = global_size[0];

    const REAL ex = extent[0];
    const REAL ey = extent[1];
    const REAL ez = extent[2];
    const REAL hex = ex*0.5;
    const REAL hey = ey*0.5;
    const REAL hez = ez*0.5;
    
    

    /*
    for( INT64 nx=0 ; nx<ntotal ; nx++){
        INT64 tcell = C[nx];
        if (tcell<0){
            printf("err: Negative cell: %d, Particle %d\n", tcell, nx);
            return -1;
        }
    }
    */
    
    // initalise the linked list
    for(INT64 llx=ll_cstart ; llx<ll_cend ; llx++){ 
        ll_array[llx] = -1; 
        ll_ccc_array[llx-ll_cstart] = 0;
    }

    // build linked list based on global cell of particle
#pragma omp parallel for default(none) reduction(min:err) reduction(max:max_cell_count) \
shared(ll_array, ll_ccc_array, C, P, global_size)
    for(INT64 nx=0 ; nx<ntotal ; nx++ ){
        INT64 _mcellc = 0;

        INT64 tcell = C[nx];
        if (tcell < 0){
            err=-4; printf("Err -4 /3: Bad particle cell %d\n", tcell);
        }

        INT64 tcx = tcell % global_size[2];
        INT64 tcy = ((tcell - tcx) / global_size[2]) % global_size[1];
        INT64 tcz = (tcell - tcx - tcy*global_size[2])/(global_size[1]*global_size[2]);
        
        if ((tcx < 0) || (tcx>=global_size[2])){
            printf("err -10: bad x cell: %d, particle %d\n", tcx, nx);
            err = -10;
        }
        if ((tcy < 0) || (tcx>=global_size[1])){
            printf("err -11: bad y cell: %d, particle %d\n", tcy, nx);
            err = -11;
        }
        if ((tcz < 0) || (tcx>=global_size[0])){
            printf("err -12: bad z cell: %d, particle %d\n", tcz, nx);
            err = -12;
        }

        tcx += 3;
        tcy += 3;
        tcz += 3;
        
        INT64 skip = 0;

        if (nx >= nlocal) {
            const REAL hpx = P[3*nx+0];
            const REAL hpy = P[3*nx+1];
            const REAL hpz = P[3*nx+2];
            if (hpx >= hex)         { tcx += hshift_x; }
            if (hpx <= -1.0*hex)    { tcx -= hshift_x; }
            if (hpy >= hey)         { tcy += hshift_y; }
            if (hpy <= -1.0*hey)    { tcy -= hshift_y; }
            if (hpz >= hez)         { tcz += hshift_z; }
            if (hpz <= -1.0*hez)    { tcz -= hshift_z; }
            
            // halo particle is very far from this domain
            if ((tcx < 0) || (tcx>=pgsx)){
                skip = 1;
                continue;
            }
            if ((tcy < 0) || (tcy>=pgsy)){
                skip = 1;
                continue;
            }
            if ((tcz < 0) || (tcz>=pgsz)){
                skip = 1;
                continue;
            }
        }

        tcell = tcx + pgsx*(tcy + tcz*pgsy);

        if (tcell < 0){
            err=-4; printf("Err -4 /2: Bad particle (nlocal, id, cell, max_cell): (%d, %d, %d, %d)\n",
                    nlocal, nx, tcell, ncells_padded);
        }

        if ((tcell < 0 || tcell >= ncells_padded ) && (err>=0)) {
            err=-4; printf("Err -4: Bad particle (nlocal, id, cell, max_cell, skip): (%d, %d, %d, %d, %d)\n", nlocal, nx, tcell, ncells_padded, skip);
        }

        if ((!(tcell < 0 || tcell >= ncells_padded )) && (err>=0)){
#pragma omp critical
            {
                  ll_array[nx] = ll_array[ll_cstart+tcell];
                  ll_array[ll_cstart+tcell] = nx;
                  ll_ccc_array[tcell]++;
                  _mcellc = MAX(_mcellc, ll_ccc_array[tcell]);
            }

        }
        max_cell_count = MAX(_mcellc, max_cell_count);
    }
    
    if (err < 0) { return err; }

    //max_cell_count += 1; //need the extra element to indicate the occupancy.

    
    *req_len = ncells_local * max_cell_count;
    *ret_max_cell_count = max_cell_count;

    err = cudaDeviceSynchronize();
    CUDACHECKERR

    err = cudaStreamDestroy(s1);
    CUDACHECKERR
    err = cudaStreamDestroy(s2);
    CUDACHECKERR
    err = cudaStreamDestroy(s3);
    CUDACHECKERR
    return err;
}


__device__ INT32 d_get_global_cell(
    const INT32 threadid
){
        const INT32 cx = threadid / d_max_cell_count;
        const INT32 cxx = cx % d_plsx;
        const INT32 cxy = ((cx - cxx) / d_plsx) % d_plsy;
        const INT32 cxz = (cx - cxx - cxy*d_plsx) / (d_plsx*d_plsy);
        // indexing tuple of "first" cell in xyz
        // -1 accounts for the padding by one cell. + 3 for the global padding
        const INT32 gxt[3] = {  cxx + d_local_offset2 -1 + 3, 
                                cxy + d_local_offset1 -1 + 3, 
                                cxz + d_local_offset0 -1 + 3};

        // global index of "first" cell
        
        return gxt[0] + d_pgsx*(gxt[1] + gxt[2]*d_pgsy);
}


__device__ inline INT32 d_get_particle_id(
    const INT32 threadid,
    const INT32 global_cell,
    const INT64 * RESTRICT d_ll_array
){  
    const INT32 cell_particle_ind = threadid % d_max_cell_count;
    INT64 cur = d_ll_array[d_ll_cstart + global_cell];
    INT32 counter = 0;
    while( (cur > -1) && (counter < cell_particle_ind)){
        cur = d_ll_array[cur];
        counter++;
    }
    return cur;
}



__global__ void other_cell(
    const INT64 * RESTRICT d_ll_array,
    const REAL * RESTRICT d_positions,
    const REAL * RESTRICT d_charges,
    REAL * RESTRICT d_forces,
    REAL * RESTRICT d_potential_array
){
    
    const INT32 threadid = threadIdx.x + blockIdx.x*blockDim.x;

    INT32 global_cell = d_get_global_cell(threadid);
    const INT32 ix = d_get_particle_id(threadid, global_cell, d_ll_array);
    global_cell += d_ll_cstart;

    if (ix < 0) {return;}
    if (ix >= d_nlocal) {return;}

    const REAL px = d_positions[ix*3    ];
    const REAL py = d_positions[ix*3 + 1];
    const REAL pz = d_positions[ix*3 + 2];


    REAL fx = 0.0;
    REAL fy = 0.0;
    REAL fz = 0.0;
    REAL uu = 0.0;

    INT32 jx = d_ll_array[global_cell];



    while( jx > -1){
        if (jx != ix){
            const REAL * RESTRICT py_ptr = &d_positions[jx*3];
            const REAL dx = *(py_ptr++) - px;
            const REAL dy = *(py_ptr++) - py;
            const REAL dz = *(py_ptr)   - pz;
            const REAL r2 = dx*dx + dy*dy + dz*dz;
            const REAL rr = sqrt(r2);
            const REAL ir = 1.0/rr;
            uu += ir * d_charges[jx];
            fx -= ir * ir * ir * d_charges[jx] * dx;
            fy -= ir * ir * ir * d_charges[jx] * dy;
            fz -= ir * ir * ir * d_charges[jx] * dz;
        }
        jx = d_ll_array[jx];
    }

    for(int ox=25 ; ox>=0 ; ox--){
        jx = d_ll_array[global_cell + d_offsets[ox]];
        while( jx > -1){
            const REAL * RESTRICT py_ptr = &d_positions[jx*3];
            const REAL dx = *(py_ptr++) - px;
            const REAL dy = *(py_ptr++) - py;
            const REAL dz = *(py_ptr)   - pz;
            const REAL r2 = dx*dx + dy*dy + dz*dz;
            const REAL rr = sqrt(r2);
            REAL ir = 1.0/rr;
            uu += ir * d_charges[jx];
            fx -= ir * ir * ir * d_charges[jx] * dx;
            fy -= ir * ir * ir * d_charges[jx] * dy;
            fz -= ir * ir * ir * d_charges[jx] * dz;

            jx = d_ll_array[jx];
        }
    }

    d_forces[ix*3 + 0] = fx * d_charges[ix];
    d_forces[ix*3 + 1] = fy * d_charges[ix];
    d_forces[ix*3 + 2] = fz * d_charges[ix];
    d_potential_array[ix] = uu * d_charges[ix];

    return;
}





extern "C"
int local_cell_by_cell_1(
    const INT64 free_space,
    const INT64 * RESTRICT global_size,
    const INT64 * RESTRICT local_size,
    const INT64 * RESTRICT local_offset,
    const INT64 num_threads,
    const INT64 nlocal,
    const INT64 ntotal,
    INT64 * RESTRICT exec_count,   
    INT64 * RESTRICT ll_array,
    INT64 * RESTRICT ll_ccc_array,
    const INT64 thread_block_size,
    const INT64 device_number,
    REAL * RESTRICT d_positions,
    REAL * RESTRICT d_charges,
    REAL * RESTRICT d_forces,
    REAL * RESTRICT d_potential_array,
    INT64 * RESTRICT d_ll_array,
    INT64 * RESTRICT d_ll_ccc_array,
    const INT64 max_cell_count,
    REAL * RESTRICT h_forces,
    REAL * RESTRICT h_potential_array
){

    const INT64 padg = 3;
    const INT64 padl = 1;

    const INT64 plsx = local_size[2] + 2*padl;
    const INT64 plsy = local_size[1] + 2*padl;
    const INT64 plsz = local_size[0] + 2*padl;

    const INT64 pgsx = global_size[2] + 2*padg;
    const INT64 pgsy = global_size[1] + 2*padg;
    const INT64 pgsz = global_size[0] + 2*padg;

    const INT64 ncells_local = plsx*plsy*plsz;

    const INT64 ncells_global = global_size[0]*global_size[1]*global_size[2];
    const INT64 ncells_padded = pgsx * pgsy * pgsz;
    const INT64 ll_cend   = ncells_padded + ntotal;
    const INT64 ll_cstart = ntotal;


    cudaStream_t s1, s2;
    int err = 0;
    dim3 thread_block(thread_block_size, 1, 1);
    
    
    err = cudaSetDevice(device_number);
    CUDACHECKERR

    int device_id = -1;
    err = cudaGetDevice(&device_id);
    CUDACHECKERR
    
    err = cudaStreamCreate(&s1);
    CUDACHECKERR
    err = cudaStreamCreate(&s2);
    CUDACHECKERR


/*
#pragma omp parallel for schedule(static, 1)
    for(INT64 cellx=0 ; cellx<ncells_padded ; cellx++){
        const INT64 _co = cellx * max_cell_count;
        INT64 _ci = 1;
        INT64 px = ll_array[ll_cstart];
        while(px>-1){
            cell_bin[_co + _ci] = px;
            _ci++;
            px = ll_array[px];
        }
        cell_bin[_co] = _ci-1; //add the occupancy to the begining
    }
*/



//    err = cudaMemcpyAsync (d_cell_bin, cell_bin,
//            ncells_padded*max_cell_count*sizeof(INT64), cudaMemcpyHostToDevice); 

    err = cudaMemcpyAsync (d_ll_array, ll_array,
            ll_cend*sizeof(INT64), cudaMemcpyHostToDevice, s1);
    CUDACHECKERR
    err = cudaMemcpyAsync (d_ll_ccc_array, ll_ccc_array,
            (ll_cend-ll_cstart)*sizeof(INT64), cudaMemcpyHostToDevice, s2);
    CUDACHECKERR

    const INT64 threads_needed = ncells_local * max_cell_count;
    const INT64 nblocks2 = ((threads_needed%thread_block_size) == 0) ? \
        threads_needed/thread_block_size : \
        threads_needed/thread_block_size + 1;

    dim3 grid_block2(nblocks2, 1, 1);

    // get offsets tuples as linear offsets
    err = cudaMemcpyToSymbol(d_max_cell_count, &max_cell_count, sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_nlocal, &nlocal, sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_ll_cstart, &ll_cstart, sizeof(INT64));
    CUDACHECKERR
    
    INT64 h_offsets[26];

    for(INT64 ox=0 ; ox<26 ; ox++){
        const INT64 hx = HMAP[ox][0] + pgsx*(HMAP[ox][1] + HMAP[ox][2]*pgsy);
        h_offsets[ox] = hx;
    }

    err = cudaMemcpyToSymbol(d_offsets, &h_offsets[0], 26*sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_plsx, &plsx, sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_plsy, &plsy, sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_pgsx, &pgsx, sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_pgsy, &pgsy, sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_local_offset2, &local_offset[2], sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_local_offset1, &local_offset[1], sizeof(INT64));
    CUDACHECKERR
    err = cudaMemcpyToSymbol(d_local_offset0, &local_offset[0], sizeof(INT64));
    CUDACHECKERR


    err = cudaDeviceSynchronize();
    CUDACHECKERR
    other_cell<<<grid_block2, thread_block>>>(d_ll_array, d_positions, d_charges, d_forces, d_potential_array);
    err = cudaDeviceSynchronize();
    CUDACHECKERR


    err = cudaMemcpyAsync(h_forces, d_forces,
            3*nlocal*sizeof(REAL), cudaMemcpyDeviceToHost, s1);
    CUDACHECKERR
    err = cudaMemcpyAsync(h_potential_array, d_potential_array,
            nlocal*sizeof(REAL), cudaMemcpyDeviceToHost, s2);
    CUDACHECKERR
    err = cudaDeviceSynchronize();
    CUDACHECKERR


    err = cudaStreamDestroy(s1);
    CUDACHECKERR
    err = cudaStreamDestroy(s2);
    CUDACHECKERR

    return err;
}






extern "C"
int local_cell_by_cell_2(
    const INT64 compute_potential,
    const INT64 nlocal,
    const INT64 device_number,
    const REAL * RESTRICT h_forces,
    const REAL * RESTRICT h_potential_array,
    REAL * RESTRICT forces,
    REAL * RESTRICT potential_array,
    REAL * U
) {

    int err = 0;
    
    //err = cudaSetDevice(device_number);
    //CUDACHECKERR

    int device_id = -1;
    //err = cudaGetDevice(&device_id);
    //CUDACHECKERR
    
    REAL uu = 0.0;

#pragma omp parallel for reduction(+:uu) default(none) \
shared(forces, potential_array, h_forces, h_potential_array) \
schedule(dynamic)
    for(INT64 px=0 ; px<nlocal ; px++){
        forces[px*3 + 0] += FORCE_UNIT * h_forces[px*3 + 0];
        forces[px*3 + 1] += FORCE_UNIT * h_forces[px*3 + 1];
        forces[px*3 + 2] += FORCE_UNIT * h_forces[px*3 + 2];
        if (compute_potential > 0){
            potential_array[px] += ENERGY_UNIT * h_potential_array[px];
        }
        uu += 0.5 * ENERGY_UNIT * h_potential_array[px];
    }
    *U = uu;
    return err;
}


