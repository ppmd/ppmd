
        #ifndef CUDA_NeighbourList
        #define CUDA_NeighbourList CUDA_NeighbourList
        
            #include <cuda_generic.h>

            extern "C" int NeighbourList(
            const int blocksize[3],
            const int threadsize[3],
            const int h_nmax,
            const int h_npart,
            const int h_nlayers,
            const double h_cutoff_squared,
            const int* __restrict__ h_CA,
            const cuda_ParticleDat<double> d_positions,
            const cuda_Array<int> d_CRL,
            const cuda_Matrix<int> d_OM,
            const cuda_Array<int> d_ccc,
            cuda_Matrix<int> d_W
            );


        #endif
        