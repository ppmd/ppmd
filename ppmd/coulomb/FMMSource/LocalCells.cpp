



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
    REAL *  RESTRICT ui,
    const INT64 *  RESTRICT ti,
    const INT64 *  RESTRICT tj
){
    REAL energy = 0.0;
    REAL * RESTRICT fiy = &fi[si];
    REAL * RESTRICT fiz = &fi[2*si]; 

    for(INT64 pxi=0 ; pxi<ni ; pxi++ ){
        REAL fx = 0.0;
        REAL fy = 0.0;
        REAL fz = 0.0;
        REAL energyi = 0.0;

        const REAL px = pi[     + pxi];
        const REAL py = pi[1*si + pxi];
        const REAL pz = pi[2*si + pxi];
        const REAL q = qi[pxi];

#pragma omp simd \
reduction(+:energyi) \
reduction(+:fx) \
reduction(+:fy) \
reduction(+:fz) \
simdlen(8)
        for(INT64 pxj=0 ; pxj<nj ; pxj++){
            const REAL dx = pj[     + pxj] - px ;
            const REAL dy = pj[1*sj + pxj] - py ;
            const REAL dz = pj[2*sj + pxj] - pz ;
         
            const REAL r2 = dx*dx + dy*dy + dz*dz;

            const REAL mask = (ti[pxi] == tj[pxj]) ? 0.0 : 1.0;
            const REAL r = sqrt(r2) + (1.0 - mask);

            const REAL ir = 1.0/r;

            const REAL term1 = q * qj[pxj] * ir * mask;
            energyi += term1;
            const REAL fcoeff = ir * ir * term1;
            fx -= fcoeff * dx;
            fy -= fcoeff * dy;
            fz -= fcoeff * dz;
        }
        
        energy += energyi;
        fi [pxi] += fx;
        fiy[pxi] += fy;
        fiz[pxi] += fz;
        ui[pxi] += energyi;
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
    REAL *  RESTRICT fi,
    REAL *  RESTRICT ui
){
    
    REAL * RESTRICT fiy = &fi[si];
    REAL * RESTRICT fiz = &fi[2*si];

    REAL energy = 0.0;
//#pragma omp simd \
//reduction(+:energy) \
//simdlen(8)        
    for(INT64 pxi=0 ; pxi<ni ; pxi++ ){
        REAL fx = 0.0;
        REAL fy = 0.0;
        REAL fz = 0.0;
        REAL energyi = 0.0;

        const REAL px = pi[     + pxi];
        const REAL py = pi[1*si + pxi];
        const REAL pz = pi[2*si + pxi];
        const REAL q = qi[pxi];
#pragma omp simd \
reduction(+:energyi) \
reduction(+:fx) \
reduction(+:fy) \
reduction(+:fz) \
simdlen(8)        
        for(INT64 pxj=0 ; pxj<nj ; pxj++){
            const REAL dx = pj[      + pxj ] - px ;
            const REAL dy = pj[ 1*sj + pxj ] - py ;
            const REAL dz = pj[ 2*sj + pxj ] - pz ;

            const REAL r2 = dx*dx + dy*dy + dz*dz;
            const REAL r = sqrt(r2);
            const REAL ir = 1.0/r;
            const REAL term1 = q * qj[pxj] * ir;
            energyi += term1;
            const REAL fcoeff = ir * ir * term1;
            fx -= fcoeff * dx;
            fy -= fcoeff * dy;
            fz -= fcoeff * dz;
        }
        energy += energyi;
        fi [pxi] += fx;
        fiy[pxi] += fy;
        fiz[pxi] += fz;
        ui[pxi] += energyi;
    }

    return energy * 0.5 * ENERGY_UNIT;
}



extern "C"
int local_cell_by_cell(
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
    REAL * RESTRICT * RESTRICT tmp_real_fi,
    REAL * RESTRICT * RESTRICT tmp_real_ui,
    const INT64 compute_potential,
    REAL * RESTRICT potential_array,
    INT64 * RESTRICT exec_count
){
    


    omp_set_num_threads(num_threads);
    int err = 0;
    REAL energy = 0.0;
    INT64 part_count = 0;
    
    // pad global and pad local
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
    
    INT64 _exec_count = 0;


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
#pragma omp parallel for default(none) reduction(min:err) \
shared(ll_array, ll_ccc_array, C, P, global_size)
    for(INT64 nx=0 ; nx<ntotal ; nx++ ){

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
            }

        }
    }
    
    if (err < 0) { return err; }

#pragma omp parallel for default(none) reduction(min:err) reduction(+:energy) reduction(+:part_count) \
reduction(+:_exec_count) \
shared(local_size, local_offset, global_size, P, Q, C, F, U, ll_array, \
ll_ccc_array, tmp_int_i, tmp_int_j, tmp_real_pi, tmp_real_pj, tmp_real_qi, \
tmp_real_qj, tmp_real_fi, tmp_real_ui, HMAP, potential_array) schedule(dynamic)
    for(INT64 cx=0 ; cx<ncells_local ; cx++ ){
        if (err < 0){ printf("Negative error code detected."); continue; }

        const INT64 threadid = omp_get_thread_num();

        INT64 * RESTRICT tmp_i    = tmp_int_i[threadid];
        INT64 * RESTRICT tmp_j    = tmp_int_j[threadid];
        REAL * RESTRICT  tmp_pi = tmp_real_pi[threadid];
        REAL * RESTRICT  tmp_pj = tmp_real_pj[threadid];
        REAL * RESTRICT  tmp_qi = tmp_real_qi[threadid];
        REAL * RESTRICT  tmp_qj = tmp_real_qj[threadid];
        REAL * RESTRICT  tmp_fi = tmp_real_fi[threadid];
        REAL * RESTRICT  tmp_ui = tmp_real_ui[threadid];

        // convert cell linear index to tuple
        // local_size is zyx
        
        const INT64 cxx = cx % plsx;
        const INT64 cxy = ((cx - cxx) / plsx) % plsy;
        const INT64 cxz = (cx - cxx - cxy*plsx) / (plsx*plsy);
        // indexing tuple of "first" cell in xyz
        // -1 accounts for the padding by one cell. + 3 for the global padding
        const INT64 gxt[3] = {  cxx + local_offset[2] -1 + 3, 
                                cxy + local_offset[1] -1 + 3, 
                                cxz + local_offset[0] -1 + 3};

        // global index of "first" cell
        const INT64 gx = gxt[0] + pgsx*(gxt[1] + gxt[2]*pgsy);
        
        // skip cell if empty
        if (ll_ccc_array[gx] == 0) { continue; }
/*
        printf("\t -- CELL -- %d %d %d ---------- %d ------------ %d %d %d\n", 
                cxx-1, cxy-1, cxz-1, gx, gxt[0], gxt[1], gxt[2]);
*/
        // populate temporary arrays
        INT64 ci_nt = 0;
        INT64 ci_ntc = 0;
        const INT64 ci_n = ll_ccc_array[gx];
        INT64 ci_tx = ll_array[ll_cstart+gx];
        while(ci_tx>-1){
            if (ci_tx > ntotal) {err=-2; printf("Err -2: Bad particle index: %d\n", ci_tx);}
            // only want to write to local particles
//            printf("gx= %d, px=%d, nt=%d, ntc=%d \n",gx, ci_tx, ci_nt, ci_ntc);
            if (ci_tx < nlocal){
//                printf("added: %f %f %f\n",P[3*ci_tx + 0], P[3*ci_tx + 1], P[3*ci_tx + 2] );
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
                // zero this particle's potential
                tmp_ui[ci_nt] = 0.0;
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
        
        // if cell contains no local particles continue
        if (ci_nt == 0) {continue;}

        for(INT64 ox=0 ; (ox<27 && (err>=0)) ; ox++){

            // global index of "second" cell as xyz tuple
            const INT64 hxtp[3] = {
                gxt[0]+HMAP[ox][0],
                gxt[1]+HMAP[ox][1],
                gxt[2]+HMAP[ox][2]
            };
            
            // may skip this cell if in free space and the cell definatly
            // cannot contain local particles
            INT64 discard_halo = 0;

            if (free_space > 0){
                if (
                    (hxtp[0] < 2) ||
                    (hxtp[1] < 2) ||
                    (hxtp[2] < 2) ||
                    (hxtp[0] > pgsx - 2 ) ||
                    (hxtp[1] > pgsy - 2 ) ||
                    (hxtp[2] > pgsz - 2 )
                ) { 
                    //continue;
                    discard_halo = 1;
                } 
                // allow drifted particles to interact with local particles
                // but not halo particles
                else if (
                    (hxtp[0] < 3) ||
                    (hxtp[1] < 3) ||
                    (hxtp[2] < 3) ||
                    (hxtp[0] > pgsx - 4 ) ||
                    (hxtp[1] > pgsy - 4 ) ||
                    (hxtp[2] > pgsz - 4 )
                ) { discard_halo = 1; }

            }

//discard_halo = 1;            
/*                
            printf("free_space %d discard %d | %d %d %d \n",
                    free_space, discard_halo, hxtp[0], hxtp[1], hxtp[2]);
            
            printf("energy: %f\n", energy);
*/
            const INT64 hxt[3] = { hxtp[0], hxtp[1], hxtp[2]};
            const INT64 hx = hxt[0] + pgsx*(hxt[1] + hxt[2]*pgsy);
            

            const INT64 cj_n = ll_ccc_array[hx];
            // if cell is empty skip
            if (cj_n == 0) { continue; }

            // populate temporary arrays
            INT64 cj_nt = 0;
            INT64 cj_ntc = 0;
            INT64 cj_tx = ll_array[ll_cstart+hx];
            
            
            while(cj_tx>-1){
                if (cj_tx > ntotal) {err=-2; printf("Err -2: Bad particle index: %d\n", cj_tx);}
                
                if ( ((discard_halo > 0) && (cj_tx<nlocal)) || (discard_halo==0) ){
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
/*
                    if (ABS(tmp_qj[cj_nt]) > 0.001){
                        printf("---\n");
                        printf("%f %f %f\n", P[3*cj_tx + 0], P[3*cj_tx + 1], P[3*cj_tx + 2]);
                        printf("%f %f %f\n", tmp_pj[cj_n*0 + cj_nt], tmp_pj[cj_n*1 + cj_nt], tmp_pj[cj_n*2 + cj_nt]);
                        printf("---\n");
                    }
*/
                    // increase particle counter
                    cj_nt++;
                }

                cj_ntc++;
                cj_tx = ll_array[cj_tx];
            }
            
            if (cj_nt == 0) { continue; }

//printf("gx %d --- dir %d | %d : %d\n", gx, ox, ci_nt, cj_nt);

            if (cj_ntc != cj_n) {err=-3; printf("Err -3: Bad particle count: %d != %d\n", ci_nt, ci_n);}

            if (hx != gx){
                energy += compute_interactions(
                    ci_n, cj_n, ci_nt, cj_nt, tmp_pi, tmp_pj, tmp_qi, tmp_qj, tmp_fi, tmp_ui);  
            } else {
                energy += compute_interactions_same_cell(
                    ci_n, cj_n, ci_nt, cj_nt, tmp_pi, tmp_pj, tmp_qi, tmp_qj, tmp_fi, tmp_ui, tmp_i, tmp_j);  
            }
            _exec_count += ci_nt*cj_nt;

        }

        // write back the new forces
        for(INT64 px=0 ; px<ci_nt ; px++){
            const INT64 idx = tmp_i[px];
            F[3*idx + 0] += FORCE_UNIT * tmp_fi[0*ci_n + px];
            F[3*idx + 1] += FORCE_UNIT * tmp_fi[1*ci_n + px];
            F[3*idx + 2] += FORCE_UNIT * tmp_fi[2*ci_n + px];
            part_count++;
        }

        if (compute_potential>0){
            for(INT64 px=0 ; px<ci_nt ; px++){
                const INT64 idx = tmp_i[px];
                potential_array[idx] += ENERGY_UNIT * tmp_ui[px];
            }
        }


    }

    if (err<0) {return err;}
    
    if (part_count != nlocal) { 
        err=-6; 
        printf("Err -6: one or more particles missed: %d, %d\n", part_count, nlocal);
    }

    U[0] = energy;
    *exec_count = _exec_count;

    return err;
}

















