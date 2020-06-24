

extern "C"
int PlainCellList(
    const REAL * RESTRICT positions,
    const INT64 npart,
    const INT64 cell_offset,
    const INT64 * RESTRICT cell_array,
    const REAL * RESTRICT inverse_cell_lengths,
    const REAL * RESTRICT local_boundary,
    INT64 * RESTRICT list,          // cell list
    INT64 * RESTRICT ccc,           // cell_contents_count
    INT64 * RESTRICT crl            // cell_reverse_lookup
){
    int err = 0;
    const INT64 ncells = cell_array[0]*cell_array[1]*cell_array[2];
#define MAX(x,y) (((x)>(y))?(x):(y))
    const INT64 max_threads = MAX(omp_get_max_threads(), 1);
    const INT64 static_size = (npart/max_threads)+1;

    // bin particles into cells
    // TODO
//#pragma omp parallel for default(none) \
//shared(positions, crl, inverse_cell_lengths, cell_array,\
//local_boundary) schedule(static, static_size)
#pragma omp parallel for schedule(static, static_size)   
    for(INT64 px=0 ; px<npart ; px++){
        const REAL rx = positions[px*3]   - local_boundary[0];
        const REAL ry = positions[px*3+1] - local_boundary[1];
        const REAL rz = positions[px*3+2] - local_boundary[2];
        
        INT64 cx = (INT64) (rx * inverse_cell_lengths[0]);
        INT64 cy = (INT64) (ry * inverse_cell_lengths[1]);
        INT64 cz = (INT64) (rz * inverse_cell_lengths[2]);
        
        INT64 cell = -1;
        if ( 
            (cx>=0) && (cx<cell_array[0]) &&
            (cy>=0) && (cy<cell_array[1]) &&
            (cz>=0) && (cz<cell_array[2])
        ) {
            cell = cx + cell_array[0]*(cy + cell_array[1]*cz);
        }
        crl[px] = cell;
    }


    // construct cell list from binned particles
    // might be quicker in serial
    // TODO
//#pragma omp parallel default(none) \
//shared(crl, ccc, cell_array, list)

#pragma omp parallel 
    {
        const INT64 nthread = omp_get_num_threads();
        const INT64 threadid = omp_get_thread_num();
        const INT64 block_size = ncells/nthread;
        const INT64 start = block_size*threadid;
        const INT64 end = (threadid == nthread-1) ? ncells : (threadid+1)*block_size;

        for(INT64 cx=start ; cx<end ; cx++){
            ccc[cx] = 0;
            list[cell_offset+cx] = -1;
        }
        
        for(INT64 px=0 ; px<npart ; px++){
            const INT64 cell_px = crl[px];
            if (( start <= cell_px ) && ( cell_px < end )){
                ccc[cell_px]++;

                // this could false share badly
                list[px] = list[cell_offset + cell_px];
                list[cell_offset + cell_px] = px;
            }
        }
    }

    
    return err;
}



