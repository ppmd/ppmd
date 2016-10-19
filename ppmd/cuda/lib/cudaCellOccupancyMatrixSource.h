
        #ifndef CUDA_CellOccupancyMatrix_947c46665fb48ffb86874b8bc9619e73_H
        #define CUDA_CellOccupancyMatrix_947c46665fb48ffb86874b8bc9619e73_H CUDA_CellOccupancyMatrix_947c46665fb48ffb86874b8bc9619e73_H
        
        //Header
        #include <cuda_generic.h>
        #include <mpi.h>

        extern "C" int LayerSort(const int f_MPI_COMM,
                     const int MPI_FLAG,
                     const int blocksize[3],
                     const int threadsize[3],
                     const int blocksize2[3],
                     const int threadsize2[3],
                     const int n,
                     const int nc,
                     int* nl,
                     int* n_cells,
                     int* __restrict__ d_pl,
                     int* __restrict__ d_crl,
                     int* __restrict__ d_ccc,
                     int** __restrict__ d_M,
                     const int* __restrict__ h_ca,
                     const double* __restrict__ h_b,
                     const double* __restrict__ h_cel,
                     const double* __restrict__ d_p
                     );
        extern "C" int copy_matrix_cols(const int h_old_ncol,
                                        const int h_new_ncol,
                                        const int h_nrow,
                                        const int * __restrict__ d_old_ptr,
                                        int * __restrict__ d_new_ptr);

        
        #endif
        