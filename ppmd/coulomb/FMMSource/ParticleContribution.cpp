
static inline INT64 compute_cell(
    const REAL* RESTRICT cube_inverse_len,
    const REAL px,
    const REAL py,
    const REAL pz,
    const REAL * RESTRICT boundary,
    const UINT64 * RESTRICT cube_offset,
    const UINT64 * RESTRICT cube_dim
){
    const REAL pxs = px + 0.5 * boundary[0];
    const REAL pys = py + 0.5 * boundary[1];
    const REAL pzs = pz + 0.5 * boundary[2];

    const UINT64 cx = ((UINT64) pxs*cube_inverse_len[0]) - cube_offset[2];
    const UINT64 cy = ((UINT64) pys*cube_inverse_len[1]) - cube_offset[1];
    const UINT64 cz = ((UINT64) pzs*cube_inverse_len[2]) - cube_offset[0];

    if (cx >= cube_dim[2] || cy >= cube_dim[1] || cz >= cube_dim[0] ){
        return (INT64) -1;}
    
    return cx + cube_dim[2] * (cy + cube_dim[1] * cz);
}


INT32 particle_contribution(
    const UINT64 npart,
    const INT32 thread_max,
    const REAL * RESTRICT position,             // xyz
    const REAL * RESTRICT boundary,             // xl. xu, yl, yu, zl, zu
    const UINT64 * RESTRICT cube_offset,        // zyx (slowest to fastest)
    const UINT64 * RESTRICT cube_dim,           // as above
    const UINT64 * RESTRICT cube_side_counts,   // as above
    REAL * RESTRICT cube_data,                  // lexicographic
    INT32 * RESTRICT thread_assign
){
    INT32 err = 0;
    omp_set_num_threads(thread_max);

    const REAL cube_ilen[3] = {
        cube_side_counts[2]/boundary[0],
        cube_side_counts[1]/boundary[1],
        cube_side_counts[0]/boundary[2]
    };

    // loop over particle and assign them to threads based on cell
    #pragma omp parallel for default(none) shared(thread_assign, position, boundary, \
     cube_offset, cube_dim, err) 
    for(UINT64 ix=0 ; ix<npart ; ix++){
        const INT64 ix_cell = compute_cell(cube_ilen, position[ix*3], position[ix*3+1], 
                position[ix*3+2], boundary, cube_offset, cube_dim);

        if (ix_cell < 0) {
            #pragma omp critical
            {err = -1;}
            #pragma omp flush (err)
        } else {
            const INT64 thread_owner = ix_cell % thread_max;
            INT64 thread_layer;
            #pragma omp critical
            {thread_layer = ++thread_assign[thread_owner];}
            thread_assign[thread_max + npart*thread_owner + thread_layer - 1] = ix;
        }
    }
 
    // check all particles were assigned to a thread
    UINT64 check_sum = 0;
    for(INT32 ix=0 ; ix<thread_max ; ix++){check_sum += thread_assign[ix];}
    if (check_sum != npart) {printf("npart %d assigned %d\n", npart, check_sum); return -2;}
    
    // Above we assigned particles to threads in a way that avoids needing atomics here.
    // Computing the cell is duplicate work, but the computation is cheap and the positions
    // needed to be read anyway to compute the moments.

    // using omp parallel for with schedule(static, 1) is robust against the omp
    // implementation deciding to launch less threads here than thread_max (which it
    // is entitled to do).
    
    UINT32 count = 0;
    #pragma omp parallel for default(none) shared(thread_assign, position, boundary, \
        cube_offset, cube_dim, err, cube_data) schedule(static,1) reduction(+: count)
    for(UINT32 tx=0 ; tx<thread_max ; tx++){
        for(UINT64 px=0 ; px<thread_assign[tx] ; px++){
            UINT64 ix = thread_assign[thread_max + npart*tx + px];

            const INT64 ix_cell = compute_cell(cube_ilen, position[ix*3], position[ix*3+1], 
                    position[ix*3+2], boundary, cube_offset, cube_dim);
            
            if (tx != ix_cell % thread_max) {           
                #pragma omp critical
                {err = -3;}
            }



            cube_data[ix_cell] += 1.0;




            count++;
        }
    }   


    
    if (count != npart) {err = -4;}
    return err;
}









