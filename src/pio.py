###############################################################################################################
# Parallel IO
###############################################################################################################


import runtime
import sys

###############################################################################################################
# pprint
###############################################################################################################

def pprint(*args):
    """
    Print a string on stdout using the default MPI handle.
    :param string:
    :return:
    """
    runtime.MPI_HANDLE.print_str(*args)

def rprint(*args):
    """
    Print a string on stdout from all procs.
    :param string:
    :return:
    """
    _s = ''
    for ix in args:
        _s += str(ix)

    for ix in range(runtime.MPI_HANDLE.nproc):
        if runtime.MPI_HANDLE.rank == ix:
            print "rank",runtime.MPI_HANDLE.rank,":",_s
            sys.stdout.flush()

        runtime.MPI_HANDLE.barrier()