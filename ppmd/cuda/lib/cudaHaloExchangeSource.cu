

__global__ void cudaPackParticleDat(
    const int d_pos_flag,
    const int d_n,
    const int d_cccmax,
    const int d_occ_m_stride,
    const int d_offset,
    const int * __restrict__ d_b_indices,
    const int * __restrict__ d_occ_matrix,
    const int * __restrict__ d_ccc,
    const int * __restrict__ d_b_scan,
    const double* __restrict__ d_shift,
    const %(DTYPE)s * __restrict__ d_ptr,
    %(DTYPE)s * __restrict__ d_buffer
){

    const int ncomp = %(NCOMP)s;
    const int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if (ix < (d_n*ncomp)){

        // cell index offset
        const int cio = ix / (ncomp * d_cccmax);
        // cell index
        const int cid = d_b_indices[cio];
        // particle layer in cell.
        const int pil = (ix/ncomp) %% d_cccmax;

        if (pil < d_ccc[cid]) {
            // particle component
            const int comp = ix %% ncomp;

            // get particle index.
            const int pid = ncomp * d_occ_matrix[d_occ_m_stride * cid + pil] + comp;

            /*
            if (cid==511){
            printf("(cid, stride, pil) %%d, %%d, %%d :ix %%d + %%d\n",
             cid,
             d_occ_m_stride,
             pil,
             d_occ_matrix[d_occ_m_stride * cid + pil],
             comp);
            }
            */


            // compute buffer index
            const int bid = ncomp * (d_b_scan[cio] - d_offset + pil) + comp;
            //const int bid = 0;

            //printf("\t(bid, pid, val, shift, d_n*ncomp) %%d, %%d, %%f, %%f, %%d\n", bid, pid, d_ptr[pid], d_shift[comp], d_n*ncomp);


            //if (bid > 599){printf("\t\t\tbid exceeded tmp (upper) dim");}
            //if (bid < 0){printf("\t\t\tbid exceeded tmp (lower) dim");}



            // copy data to buffer
            d_buffer[bid] = d_ptr[pid];

            // apply periodic boundary flag to packed particles
            if ( d_pos_flag == 1 ){
                d_buffer[bid] += d_shift[comp];
            }

        }

    }
    return;
}



int cudaHaloExchangePD(
    const int f_MPI_COMM,
    const int n_local,
    const int h_pos_flag,
    const int h_cccmax,
    const int h_occ_m_stride,
    const int* __restrict__ h_b_ind,
    const int* __restrict__ h_send_counts,
    const int* __restrict__ h_recv_counts,
    const int* __restrict__ SEND_RANKS,
    const int* __restrict__ RECV_RANKS,
    const int* __restrict__ d_b_indices,
    const int* __restrict__ d_occ_matrix,
    const int* __restrict__ d_ccc,
    const int* __restrict__ d_b_scan,
    const double* __restrict__ d_shift,
    %(DTYPE)s * __restrict__ d_ptr,
    %(DTYPE)s * __restrict__ d_buffer
){






    // get mpi comm and rank
    MPI_Comm MPI_COMM = MPI_Comm_f2c(f_MPI_COMM);
    int rank = -1; MPI_Comm_rank( MPI_COMM, &rank );
    MPI_Status MPI_STATUS;
    MPI_Request sr;
    MPI_Request rr;

    int DAT_END = n_local;
    int offset = 0;

    dim3 bs, ts;
    cudaError_t err;


    //cout << "rank " << rank << endl;

    // ---
    /*
    int tmp;
    int err2 = cudaMemcpy(&tmp, d_occ_matrix+511, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "511 BEFORE RUN: " << tmp << " err " << err2 << endl;

    if (rank==0){

    cout << f_MPI_COMM << endl;
    cout << n_local << endl;
    cout << h_pos_flag << endl;
    cout << h_cccmax << endl;
    cout << h_occ_m_stride << endl;

    for( int dir=0 ; dir<6 ; dir++ ){
    cout << "# " <<  dir << " ---- " << endl;
    cout << h_b_ind[dir] << endl;
    cout << SEND_RANKS[dir] << endl;
    cout << RECV_RANKS[dir] << endl;
    cout << h_send_counts[dir] << endl;
    cout << h_recv_counts[dir] << endl;
    }


    }

    */

    // ---


    for( int dir=0 ; dir<6 ; dir++ ){

        int b_s = h_b_ind[dir];
        const int cell_count = h_b_ind[dir+1] - b_s;


        const int scount = h_send_counts[dir];


        err = cudaCreateLaunchArgs(   cell_count*h_cccmax*%(NCOMP)s    , 256, &bs, &ts);
        if (err != cudaSuccess) { return err; }

        /*
        if (rank==0){
        cout << "scount " << scount << " cell_count " << cell_count << endl;
        cout << "bs.x " << bs.x << " ts.x " << ts.x << endl;
        }
        */


        cudaPackParticleDat<<<bs,ts>>>(
            h_pos_flag,
            cell_count*h_cccmax,
            h_cccmax,
            h_occ_m_stride,
            offset,
            d_b_indices + b_s,
            d_occ_matrix,
            d_ccc,
            d_b_scan + b_s,
            d_shift + dir*3,
            d_ptr,
            d_buffer
        );

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            //cout << "Error on cudaSync: " << rank << endl;
            return err;
         }


        if (( SEND_RANKS[dir] > -1 ) && ( h_send_counts[dir] > 0 ) ){
        MPI_Isend((void *) d_buffer, h_send_counts[dir] * %(NCOMP)s, %(MPI_DTYPE)s,
                 SEND_RANKS[dir], rank, MPI_COMM, &sr);
        }

        if (( RECV_RANKS[dir] > -1 ) && ( h_recv_counts[dir] > 0 ) ){
        MPI_Irecv((void *) &d_ptr[DAT_END * %(NCOMP)s], %(NCOMP)s * h_recv_counts[dir],
                  %(MPI_DTYPE)s, RECV_RANKS[dir], RECV_RANKS[dir], MPI_COMM, &rr);
        }


        // after send has completed move to next direction.
        if (( SEND_RANKS[dir] > -1 ) && ( h_send_counts[dir] > 0 ) ){
            MPI_Wait(&sr, &MPI_STATUS);
        }

        if (( RECV_RANKS[dir] > -1 ) && ( h_recv_counts[dir] > 0 ) ){
            MPI_Wait(&rr, &MPI_STATUS);
        }

        MPI_Barrier(MPI_COMM);
        DAT_END += h_recv_counts[dir];
        offset += h_send_counts[dir];

        /*
        if (rank==0){
        cout << "IL " << h_recv_counts[dir] << " " << h_send_counts[dir] << endl;
        }
        */

    }

    return err;
}