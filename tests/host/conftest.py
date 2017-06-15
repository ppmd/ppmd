
from ppmd.mpi import MPI

def pytest_report_header(config):
    if config.option.verbose > 0:
            return ["RANK: {}".format(MPI.COMM_WORLD.Get_rank())]

