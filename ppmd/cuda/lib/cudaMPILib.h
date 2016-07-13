#include <helper_cuda.h>
#include <iostream>
#include <mpi.h>

using namespace std;

// Directly related to MPI
extern "C" int MPIErrorCheck_cuda(const int error_code);
extern "C" int MPI_Bcast_cuda(const int FCOMM,
                              void* buffer,
                              const int byte_count,
                              const int root
                              );


extern "C" int MPI_Gatherv_cuda(const int FCOMM,
                                const void* s_buffer,
                                const int s_count,
                                void* r_buffer,
                                const int* r_counts,
                                const int* r_disps,
                                const int root
                                );










// MPI related multigpu static libs
extern "C" int cudaFindEmptySlots(const int blocksize1[3],
                                  const int threadsize1[3],
                                  const int* d_scan,
                                  const int h_n1,
                                  int * d_empties
                                  );

extern "C" int cudaFindNewSlots(const int blocksize2[3],
                                const int threadsize2[3],
                                const int* d_scan,
                                const int h_n1,
                                const int h_n2,
                                int * d_sources
                                );
