#system level imports
import ctypes

#package level imports
from ppmd import runtime, pio, mpi


#cuda level imports
import cuda_build
import cuda_runtime


try:
    LIB_CUDA_MPI = ctypes.cdll.LoadLibrary(cuda_build.build_static_libs('cudaMPILib'))
except:
    raise RuntimeError('cuda_mpi error: Module is not initialised correctly, CUDA MPI lib not loaded')



###############################################################################
# CUDA_MPI_err_checking
###############################################################################

def cuda_mpi_err_check(err_code):
    """
    Wrapper to check cuda error codes.
    :param err_code:
    :return:
    """

    assert LIB_CUDA_MPI is not None, "cuda_mpi error: No cudaMPILib"

    if type(err_code) is not ctypes.c_int:
        err_code = ctypes.c_int(err_code)


    err = LIB_CUDA_MPI['MPIErrorCheck_cuda'](err_code)
    assert err == 0, "Non-zero CUDA MPI error:" + str(err_code)




###############################################################################
# MPI_Bcast
###############################################################################


def MPI_Bcast(buffer, byte_count, root):
    """
    Perform MPI_Bcast with cuda pointers. All parameters are ctypes types.
    :param buffer: ctypes pointer to data
    :param byte_count: number of bytes to transfer
    :param root: root mpi rank
    """

    assert type(byte_count) is ctypes.c_int, "byte_count has incorrect data type"
    assert type(root) is ctypes.c_int, "root arg has incorrect data type"

    cuda_mpi_err_check(LIB_CUDA_MPI['MPI_Bcast_cuda'](
        ctypes.c_int(mpi.MPI_HANDLE.comm.py2f()),
        buffer,
        byte_count,
        root
    ))







































