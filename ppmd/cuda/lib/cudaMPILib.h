#include <helper_cuda.h>
#include <iostream>
#include <mpi.h>

using namespace std;

extern "C" int MPIErrorCheck_cuda(const int error_code);
extern "C" int MPI_Bcast_cuda(const int FCOMM, void* buffer, const int byte_count, const int root);





