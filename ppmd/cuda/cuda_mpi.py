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



#####################################################################################
# CUDA_MPI_err_checking
#####################################################################################

def cuda_mpi_err_check(err_code):
    """
    Wrapper to check cuda error codes.
    :param err_code:
    :return:
    """

    assert LIB_CUDA_MPI is not None, "cuda_mpi error: No cudaMPILib"

    err = LIB_CUDA_MPI['MPIErrorCheck_cuda'](err_code)
    assert err == 0, "Non-zero CUDA MPI error:" + str(err_code)


