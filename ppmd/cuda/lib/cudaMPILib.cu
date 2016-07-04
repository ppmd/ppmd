#include "cudaMPILib.h"




int MPIErrorCheck_cuda(const int error_code){

    int err = 0;
    if (error_code != MPI_SUCCESS) {

       char error_string[BUFSIZ];
       int length_of_error_string;

       MPI_Error_string(error_code, error_string, &length_of_error_string);
       //fprintf(stderr, "%3d: %s\n", my_rank, error_string);
       cout << error_string << endl;

       err = 1;
    }

    return err;
}





int MPI_Bcast_cuda(const int FCOMM, void* buffer, const int byte_count, const int root){
    MPI_Comm COMM = MPI_Comm_f2c(FCOMM);

    MPI_Errhandler_set(COMM, MPI_ERRORS_RETURN);
    const int err = MPI_Bcast( buffer,
                               byte_count,
                               MPI_BYTE,
                               root,
                               COMM
                             );

    return err;
}


