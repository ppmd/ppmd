



static inline REAL compute_interactions_same_cell(
    const INT64 si,
    const INT64 sj,
    const INT64 ni,
    const INT64 nj,
    const REAL *  RESTRICT pi,
    const REAL *  RESTRICT pj,
    const REAL *  RESTRICT qi,
    const REAL *  RESTRICT qj,
    REAL *  RESTRICT fi,
    const INT64 *  RESTRICT ti,
    const INT64 *  RESTRICT tj
){
    REAL energy = 0.0;
    for(INT64 pxi=0 ; pxi<ni ; pxi++ ){
        REAL fx = 0.0;
        REAL fy = 0.0;
        REAL fz = 0.0;

        const REAL px = pi[     + pxi];
        const REAL py = pi[1*si + pxi];
        const REAL pz = pi[2*si + pxi];
        const REAL q = qi[pxi];

        for(INT64 pxj=0 ; pxj<nj ; pxj++){
            const REAL dx = pj[     + pxj] - px ;
            const REAL dy = pj[1*si + pxj] - py ;
            const REAL dz = pj[2*si + pxj] - pz ;

            const REAL r2 = dx*dx + dy*dy + dz*dz;


            const REAL mask = (ti[pxi] == tj[pxj]) ? 0.0 : 1.0;
            const REAL r = sqrt(r2) + (1.0 - mask);

            const REAL ir = 1.0/r;


            const REAL term1 = q * qj[pxj] * ir * mask;
            energy += term1;
            const REAL fcoeff = FORCE_UNIT * ir * ir * term1;
            fx -= fcoeff * dx;
            fy -= fcoeff * dy;
            fz -= fcoeff * dz;
        }

        fi[     + pxi] += fx;
        fi[1*si + pxi] += fy;
        fi[2*si + pxi] += fz;
    }

    return energy * 0.5 * ENERGY_UNIT;
}

static inline REAL compute_interactions(
    const INT64 si,
    const INT64 sj,
    const INT64 ni,
    const INT64 nj,
    const REAL *  RESTRICT pi,
    const REAL *  RESTRICT pj,
    const REAL *  RESTRICT qi,
    const REAL *  RESTRICT qj,
    REAL *  RESTRICT fi
){
    REAL energy = 0.0;
    for(INT64 pxi=0 ; pxi<ni ; pxi++ ){
        REAL fx = 0.0;
        REAL fy = 0.0;
        REAL fz = 0.0;

        const REAL px = pi[     + pxi];
        const REAL py = pi[1*si + pxi];
        const REAL pz = pi[2*si + pxi];
        const REAL q = qi[pxi];

        for(INT64 pxj=0 ; pxj<nj ; pxj++){
            const REAL dx = pj[     + pxj] - px ;
            const REAL dy = pj[1*si + pxj] - py ;
            const REAL dz = pj[2*si + pxj] - pz ;
            const REAL r2 = dx*dx + dy*dy + dz*dz;
            const REAL r = sqrt(r2);
            const REAL ir = 1.0/r;
            const REAL term1 = q * qj[pxj] * ir;
            energy += term1;
            const REAL fcoeff = FORCE_UNIT * ir * ir * term1;
            fx -= fcoeff * dx;
            fy -= fcoeff * dy;
            fz -= fcoeff * dz;
        }

        fi[     + pxi] += fx;
        fi[1*si + pxi] += fy;
        fi[2*si + pxi] += fz;
    }

    return energy * 0.5 * ENERGY_UNIT;
}



extern "C"
int local_cell_by_cell(
    const INT64 free_space,
    const INT64 * RESTRICT global_size,
    const INT64 * RESTRICT local_size,
    const INT64 * RESTRICT local_offset,
    const INT64 num_threads,
    const INT64 nlocal,
    const INT64 ntotal,
    const REAL * RESTRICT P,
    const REAL * RESTRICT Q,
    const REAL * RESTRICT C,
    REAL * RESTRICT F,
    REAL * RESTRICT U,
    INT64 * RESTRICT ll_array,
    INT64 * RESTRICT ll_ccc_array,
    INT64 * RESTRICT * RESTRICT tmp_int_i,
    INT64 * RESTRICT * RESTRICT tmp_int_j,
    REAL * RESTRICT * RESTRICT tmp_real_pi,
    REAL * RESTRICT * RESTRICT tmp_real_pj,
    REAL * RESTRICT * RESTRICT tmp_real_qi,
    REAL * RESTRICT * RESTRICT tmp_real_qj,
    REAL * RESTRICT * RESTRICT tmp_real_fi
){

    omp_set_num_threads(num_threads);
    int err = 0;
    REAL energy = 0.0;

    const INT64 ncells_global = global_size[0]*global_size[1]*global_size[2];
    const INT64 ncells_local = local_size[0]*local_size[1]*local_size[2];

    const INT64 ll_cend   = ncells_global + ntotal;
    const INT64 ll_cstart = ntotal;
    
    // initalise the linked list
    for(INT64 llx=ll_cstart ; llx<ll_cend ; llx++){ 
        ll_array[llx] = -1; 
        ll_ccc_array[llx-ll_cstart] = 0;
    }

    // build linked list based on global cell of particle
#pragma omp parallel for default(none) shared(ll_array, ll_ccc_array, C)
    for(INT64 nx=0 ; nx<ntotal ; nx++ ){
        const INT64 tcell = C[nx];
#pragma omp critical
        {
            const INT64 tcurr = ll_array[ll_cstart+tcell];
            ll_array[ll_cstart+tcell] = nx;
            ll_array[nx] = tcurr;
            ll_ccc_array[tcell]++;
        }
    }
    
#pragma omp parallel for default(none) reduction(min:err) reduction(+:energy)\
shared(local_size, local_offset, global_size, P, Q, C, F, U, ll_array, \
ll_ccc_array, tmp_int_i, tmp_int_j, tmp_real_pi, tmp_real_pj, tmp_real_qi, \
tmp_real_qj, tmp_real_fi)
    for(INT64 cx=0 ; cx<ncells_local ; cx++ ){
        const INT64 threadid = omp_get_thread_num();

        INT64 * RESTRICT tmp_i    = tmp_int_i[threadid];
        INT64 * RESTRICT tmp_j    = tmp_int_j[threadid];
        REAL * RESTRICT  tmp_pi = tmp_real_pi[threadid];
        REAL * RESTRICT  tmp_pj = tmp_real_pj[threadid];
        REAL * RESTRICT  tmp_qi = tmp_real_qi[threadid];
        REAL * RESTRICT  tmp_qj = tmp_real_qj[threadid];
        REAL * RESTRICT  tmp_fi = tmp_real_fi[threadid];

        // convert cell linear index to tuple
        // local_size is zyx
        const INT64 cxx = cx % local_size[2];
        const INT64 cxy = ((cx - cxx) / local_size[2]) % local_size[1];
        const INT64 cxz = (cx - cxx - cxy*local_size[2]) / (local_size[2]*local_size[1]);
        // indexing tuple of "first" cell in xyz
        const INT64 gxt[3] = {cxx + local_offset[2], cxy + local_offset[1], cxz + local_offset[0]};
        // global index of "first" cell
        const INT64 gx = gxt[0] + global_size[2]*(gxt[1] + gxt[2]*global_size[1]);


        // populate temporary arrays
        INT64 ci_nt = 0;
        INT64 ci_ntc = 0;
        const INT64 ci_n = ll_ccc_array[gx];
        INT64 ci_tx = ll_array[ll_cstart+gx];
        while(ci_tx>-1){
            if (ci_tx > ntotal) {err=-2; printf("Err -2: Bad particle index: %d\n", ci_tx);}
            // only want to write to local particles
            if (ci_tx < nlocal){
                // copy positions
                tmp_pi[ci_n*0 + ci_nt] = P[3*ci_tx + 0];
                tmp_pi[ci_n*1 + ci_nt] = P[3*ci_tx + 1];
                tmp_pi[ci_n*2 + ci_nt] = P[3*ci_tx + 2];
                // copy charges
                tmp_qi[ci_nt] = Q[ci_tx];
                // zero forces
                tmp_fi[ci_n*0 + ci_nt] = 0.0;
                tmp_fi[ci_n*1 + ci_nt] = 0.0;
                tmp_fi[ci_n*2 + ci_nt] = 0.0;
                // copy particle id
                tmp_i[ci_nt] = ci_tx;
                // increase particle counter
                ci_nt++;
            }
            ci_ntc++;
            ci_tx = ll_array[ci_tx];
        }
        // failure to find all the particles
        if (ci_ntc != ci_n) {err=-1; printf("Err -1: Bad particle count: %d != %d\n", ci_nt, ci_n);}

        for(INT32 ox=0 ; ox<27 ; ox++){
            // global index of "second" cell as xyz tuple
            const INT64 hxtp[3] = {
                gxt[0]+HMAP[ox][0],
                gxt[1]+HMAP[ox][1],
                gxt[2]+HMAP[ox][2]
            };
            
            // may skip this cell if in free space
            if (free_space > 0){
                if (
                    (hxtp[0] < 0) ||
                    (hxtp[1] < 0) ||
                    (hxtp[2] < 0) ||
                    (hxtp[0] > global_size[2]) ||
                    (hxtp[1] > global_size[1]) ||
                    (hxtp[2] > global_size[0])
                ){continue;}
            }

            const INT64 hxt[3] = {
                hxtp[0] % global_size[2],
                hxtp[1] % global_size[1],
                hxtp[2] % global_size[0]
            };            

            const INT64 hx = hxt[0] + global_size[2]*(hxt[1] + hxt[2]*global_size[1]);
            

            // populate temporary arrays
            INT64 cj_nt = 0;
            INT64 cj_ntc = 0;
            const INT64 cj_n = ll_ccc_array[hx];
            INT64 cj_tx = ll_array[ll_cstart+hx];
            
            
            while(cj_tx>-1){
                if (cj_tx > ntotal) {err=-2; printf("Err -2: Bad particle index: %d\n", cj_tx);}

                // copy positions
                tmp_pj[cj_n*0 + cj_nt] = P[3*cj_tx + 0];
                tmp_pj[cj_n*1 + cj_nt] = P[3*cj_tx + 1];
                tmp_pj[cj_n*2 + cj_nt] = P[3*cj_tx + 2];
                // copy charges
                tmp_qj[cj_nt] = Q[cj_tx];
                // copy particle id
                if (hx==gx){
                    tmp_j[cj_nt] = cj_tx;
                }
                // increase particle counter
                cj_nt++;

                cj_tx = ll_array[cj_tx];
            }
            if (cj_nt != cj_n) {err=-3; printf("Err -3: Bad particle count: %d != %d\n", ci_nt, ci_n);}

            if (hx != gx){
                energy += compute_interactions(ci_n, cj_n, ci_nt, cj_nt, tmp_pi, tmp_pj,
                    tmp_qi, tmp_qj, tmp_fi);  
            } else {
                energy += compute_interactions_same_cell(ci_n, cj_n, ci_nt, cj_nt, tmp_pi, 
                tmp_pj, tmp_qi, tmp_qj, tmp_fi, tmp_i, tmp_j);  
            }


        }

        // write back the new forces
        for(INT64 px=0 ; px<ci_nt ; px++){
            const INT64 idx = tmp_i[px];
            F[3*idx + 0] += tmp_fi[0*ci_n + px];
            F[3*idx + 1] += tmp_fi[1*ci_n + px];
            F[3*idx + 2] += tmp_fi[2*ci_n + px];
        }

    }

    U[0] = energy;

    return err;
}

















