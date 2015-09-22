###############################################################################################################
# Parallel IO helper functions.
###############################################################################################################


import mpi
import sys

###############################################################################################################
# pprint
###############################################################################################################

def pprint(*args):
    """
    Print a string on stdout using the default MPI handle using rank 0.
    :param string:
    :return:
    """
    mpi.MPI_HANDLE.print_str(*args)

def rprint(*args):
    """
    Print a string on stdout from all procs. preappended with rank id.
    :param string:
    :return:
    """

    _s = ''
    for ix in args:
        _s += str(ix)

    for ix in range(mpi.MPI_HANDLE.nproc):
        if mpi.MPI_HANDLE.rank == ix:
            print "rank",mpi.MPI_HANDLE.rank,":",_s
            sys.stdout.flush()

        mpi.MPI_HANDLE.barrier()
