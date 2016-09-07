

#include "cudaMoveLib.h"





int cudaMoveStageOne(
    const int FCOMM,
    const int * h_send_ranks,
    const int * h_recv_rank,
    int * h_send_counts,
    int * h_recv_counts
){
    MPI_Comm COMM = MPI_Comm_f2c(FCOMM);
    MPI_Errhandler_set(COMM, MPI_ERRORS_RETURN);
    int rank; MPI_Comm_rank(COMM, &rank);
    int err = 0;

    MPI_Request SR[26];
    MPI_Request RR[26];
    MPI_Status SS[26];
    MPI_Status RS[26];

    int src = 0;

    for(int dir=0 ; dir<26 ; dir++){

        if ( (rank == h_send_ranks[dir]) && (rank == h_recv_rank[dir]) ){
            // This send should not occur has it would send to itself.
            // Send count is also zeroed to reduce memory footprint.
            h_recv_counts[dir] = 0;
            h_send_counts[dir] = 0;
        } else {
            err |= MPI_Isend(&h_send_counts[dir],
                             1,
                             MPI_INT,
                             h_send_ranks[dir],
                             rank,
                             COMM,
                             &SR[src]);


            err |= MPI_Irecv(&h_recv_counts[dir],
                             1,
                             MPI_INT,
                             h_recv_rank[dir],
                             h_recv_rank[dir],
                             COMM,
                             &RR[src]);
            src++;
        }
    }

    if (err>0) {return err;}


    err = MPI_Waitall(src, SR, SS);
    if (err>0) {return err;}
    err = MPI_Waitall(src, RR, RS);

    return err;
}





__global__ void d_pack(
    const int count,
    const int total_bytes,
    const int num_dats,
    const int * __restrict__ d_byte_counts,
    const int * __restrict__ d_move_matrix,
    void ** d_ptrs,
    char * __restrict__ d_buffer,
    int * __restrict__ d_empty_flag
){
    const int tx = threadIdx.x + blockIdx.x*blockDim.x;
    if( tx<count ){

        // particle id to pack
        const int px = d_move_matrix[tx];
        // starting offset in d_buffer
        const int offset = total_bytes*tx;

        int loffset=0;
        for(int dat=0 ; dat<num_dats ; dat++){

            memcpy(d_buffer+offset+loffset,
                   ((char*) d_ptrs[dat])+px*d_byte_counts[dat],
                   d_byte_counts[dat]
                   );

            loffset += d_byte_counts[dat];
        }
        d_empty_flag[px] = 1;
    }

    return;
}

__global__ void d_unpack(
    const int d_n,
    const int thread_count,
    const int total_bytes,
    const int num_dats,
    const int * __restrict__ d_byte_counts,
    void ** d_ptrs,
    const char * __restrict__ d_buffer
){
    const int tx = threadIdx.x + blockIdx.x*blockDim.x;
    if( tx<thread_count ){
        // buffer particle index
        const int pxn = tx/total_bytes;
        // dat index to copy into
        const int pxd = pxn+d_n;

        int buf_off = 0;
        for( int dx=0 ; dx<num_dats; dx++){
            const int bc = d_byte_counts[dx];
            memcpy(
                ((char*)d_ptrs[dx]) + pxd*bc,
                d_buffer + total_bytes*pxn + buf_off,
                bc
            );

            buf_off += bc;
        }
    }

    return;
}

int cudaMoveStageTwo(
    const int FCOMM,
    const int n_local,
    const int total_bytes,
    const int num_dats,
    const int * __restrict__ h_send_counts,
    const int * __restrict__ h_recv_counts,
    const int * __restrict__ h_send_ranks,
    const int * __restrict__ h_recv_ranks,
    const int * __restrict__ d_move_matrix,
    const int move_matrix_stride,
    char * __restrict__ d_send_buf,
    char * __restrict__ d_recv_buf,
    void ** h_ptrs,
    const int * __restrict__ h_byte_counts,
    int * __restrict__ d_empty_flag
){

    cudaError_t err;
    int err_mpi = 0;

    dim3 bs, ts;


    int * d_byte_counts;
    err = cudaMalloc(&d_byte_counts, num_dats*sizeof(int));
    if(err>0){return err;}
    err = cudaMemcpy(d_byte_counts,
                     h_byte_counts,
                     num_dats*sizeof(int),
                     cudaMemcpyHostToDevice);
    if(err>0){return err;}

    void ** d_ptrs;
    err = cudaMalloc(&d_ptrs, num_dats*sizeof(void*));
    if(err>0){return err;}
    err = cudaMemcpy(d_ptrs,
                     h_ptrs,
                     num_dats*sizeof(void*),
                     cudaMemcpyHostToDevice);
    if(err>0){return err;}



    MPI_Comm COMM = MPI_Comm_f2c(FCOMM);
    MPI_Errhandler_set(COMM, MPI_ERRORS_RETURN);
    int rank; MPI_Comm_rank(COMM, &rank);

    MPI_Request SR[26];
    MPI_Request RR[26];
    MPI_Status SS[26];
    MPI_Status RS[26];


    // PACKING ---------------------------------------
    int s_buffer_offset = 0;
    for(int dir=0 ; dir<26 ; dir++ ){
        if ( ! ((rank == h_send_ranks[dir]) && (rank == h_recv_ranks[dir])) ){

            err = cudaCreateLaunchArgs(h_send_counts[dir], 256, &bs, &ts);
            if(err>0){return err;}

            d_pack<<<bs,ts>>>(
                h_send_counts[dir],
                total_bytes,
                num_dats,
                d_byte_counts,
                d_move_matrix+dir*move_matrix_stride,
                d_ptrs,
                d_send_buf + s_buffer_offset,
                d_empty_flag
            );

            s_buffer_offset += h_send_counts[dir]*total_bytes;
        }
    }


    err = cudaDeviceSynchronize();
    if(err>0){return err;}


    // SENDING ---------------------------------------
    s_buffer_offset = 0;
    int r_buffer_offset = 0;
    int src = 0;


    for(int dir=0 ; dir<26 ; dir++){

        if ( ! ((rank == h_send_ranks[dir]) && (rank == h_recv_ranks[dir])) ){
            err_mpi = err_mpi | MPI_Isend(d_send_buf+s_buffer_offset,
                             h_send_counts[dir]*total_bytes,
                             MPI_BYTE,
                             h_send_ranks[dir],
                             rank,
                             COMM,
                             &SR[src]);


            err_mpi = err_mpi | MPI_Irecv(d_recv_buf+r_buffer_offset,
                             h_recv_counts[dir]*total_bytes,
                             MPI_BYTE,
                             h_recv_ranks[dir],
                             h_recv_ranks[dir],
                             COMM,
                             &RR[src]);
            src++;
            s_buffer_offset += h_send_counts[dir]*total_bytes;
            r_buffer_offset += h_recv_counts[dir]*total_bytes;
        }
    }

    if (err_mpi>0) {return err_mpi;}


    err_mpi = MPI_Waitall(src, SR, SS);
    if (err_mpi>0) {return err_mpi;}
    err_mpi = MPI_Waitall(src, RR, RS);
    if (err_mpi>0) {return err_mpi;}

    //UNPACKING ---------------------------------------

    err = cudaCreateLaunchArgs(r_buffer_offset, 256, &bs, &ts);
    if(err>0){return err;}

    d_unpack<<<bs,ts>>>(
        n_local,
        r_buffer_offset,
        total_bytes,
        num_dats,
        d_byte_counts,
        d_ptrs,
        d_recv_buf
    );
    err = cudaDeviceSynchronize();

    return err;
}









