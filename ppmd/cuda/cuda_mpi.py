#system level imports
import ctypes

#package level imports


#cuda level imports
import cuda_build


try:
    LIB_CUDA_MPI = ctypes.cdll.LoadLibrary(cuda_build.build_static_libs('cudaMPILib'))
except Exception as e:
    raise e
    # raise RuntimeError('cuda_mpi error: Module is not initialised correctly,'
    #                   ' CUDA MPI lib not loaded\n')



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


def MPI_Bcast(comm, buffer, byte_count, root):
    """
    Perform MPI_Bcast with cuda pointers. All parameters are ctypes types.
    :param comm: mpi4py MPI communicator to use
    :param buffer: ctypes pointer to data
    :param byte_count: number of bytes to transfer
    :param root: root mpi rank
    """

    assert type(byte_count) is ctypes.c_int, "byte_count has incorrect data type"
    assert type(root) is ctypes.c_int, "root arg has incorrect data type"

    cuda_mpi_err_check(LIB_CUDA_MPI['MPI_Bcast_cuda'](
        ctypes.c_int(comm.py2f()),
        buffer,
        byte_count,
        root
    ))


###############################################################################
# MPI_Gatherv
###############################################################################


def MPI_Gatherv(comm,s_buffer, s_count, r_buffer, r_counts, r_disps, root):
    """
    Cuda aware MPI_Gatherv. All counts/sizes are in bytes not counts.
    :param comm: mpi4py MPI communicator to use
    :param s_buffer: ctypes pointer to send buffer
    :param s_count: ctypes.c_int of send count
    :param r_buffer: ctypes pointer to recv buffer
    :param r_counts: ctypes pointer to int array of recv counts
    :param r_disps: ctypes pointer to int array of recv offsets
    :param root: ctypes.c_int root process
    :return:
    """

    cuda_mpi_err_check(LIB_CUDA_MPI['MPI_Gatherv_cuda'](
        ctypes.c_int(comm.py2f()),
        s_buffer,
        s_count,
        r_buffer,
        r_counts,
        r_disps,
        root
    ))


def cuda_exclusive_scan_int_masked_copy(length, d_map, d_ccc, d_scan):
    m = ctypes.c_int32(0)
    LIB_CUDA_MPI['cudaHaloArrayCopyScan'](
        ctypes.c_int32(length),
        d_map.ctypes_data,
        d_ccc.ctypes_data,
        d_scan.ctypes_data,
        ctypes.byref(m)
    )
    return m.value

































