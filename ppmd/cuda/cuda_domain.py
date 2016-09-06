

# system level
import numpy as np
import ctypes

# package level
from ppmd import opt, host, mpi, kernel, access


# cuda level
import cuda_runtime
import cuda_loop
import cuda_base
import cuda_build


class BoundaryTypePeriodic(object):
    """
    Class to hold and perform periodic boundary conditions.

    :arg state_in: State on which to apply periodic boundaries to.
    """

    def __init__(self, state_in=None):
        self.state = state_in

        # Initialise timers
        self.timer_apply = opt.Timer(cuda_runtime.TIMER, 0)
        self.timer_lib_overhead = opt.Timer(cuda_runtime.TIMER, 0)
        self.timer_search = opt.Timer(cuda_runtime.TIMER, 0)
        self.timer_move = opt.Timer(cuda_runtime.TIMER, 0)

        # One proc PBC lib
        self._one_process_pbc_lib = None
        # Escape guard lib
        self._escape_guard_lib = None
        self._escape_count = None
        self._escape_dir_count = None
        self._escape_list = None
        self._escape_matrix = None

        self._flag = cuda_base.Array(ncomp=1, dtype=ctypes.c_int)

    def set_state(self, state_in=None):
        assert state_in is not None, "BoundaryTypePeriodic error: No state" \
                                     " passed."
        self.state = state_in

    def apply(self):
        """
        Enforce the boundary conditions on the held state.
        """

        self.timer_apply.start()

        self._flag[0] = 0

        if mpi.MPI_HANDLE.nproc == 1:
            """
            BC code for one proc. porbably removable when restricting to large
             parallel systems.
            """

            self.timer_lib_overhead.start()

            if self._one_process_pbc_lib is None:
                with open(str(cuda_runtime.LIB_DIR) +
                                  '/cudaOneProcPBCSource.cu','r') as fh:
                    _one_proc_pbc_code = fh.read()

                _one_proc_pbc_kernel = kernel.Kernel(
                    '_one_proc_pbc_kernel',
                    _one_proc_pbc_code,
                    None,
                    static_args={'E0':ctypes.c_double,
                                 'E1':ctypes.c_double,
                                 'E2':ctypes.c_double}
                    )

                self._one_process_pbc_lib = cuda_loop.ParticleLoop(
                    _one_proc_pbc_kernel,
                    {'P': self.state.get_position_dat()(access.RW),
                     'BCFLAG':self._flag}
                )


            self.timer_lib_overhead.pause()

            _E = self.state.domain.extent

            self.timer_move.start()
            self._one_process_pbc_lib.execute(
                n=self.state.get_position_dat().npart_local,
                static_args={'E0':ctypes.c_double(_E[0]),
                             'E1':ctypes.c_double(_E[1]),
                             'E2':ctypes.c_double(_E[2])}
            )
            self.timer_move.pause()


        ############ ----- MULTIPROC -------
        else:
            if self._escape_guard_lib is None:
                # build lib
                self._escape_guard_lib = \
                    cuda_build.build_static_libs('cudaNProcPBC')

            # --- init escape count ----
            if self._escape_count is None:
                self._escape_count = cuda_base.Array(ncomp=1,
                                                     dtype=ctypes.c_int32)
            self._escape_count[0] = 0

            # --- init escape dir count ----
            if self._escape_dir_count is None:
                self._escape_dir_count = cuda_base.Array(ncomp=26,
                                                         dtype=ctypes.c_int32)
            self._escape_dir_count[:] = 0


            # --- init escape list ----
            nl3 = self.state.get_position_dat().npart_local * 3

            if self._escape_list is None:
                self._escape_list = cuda_base.Array(
                    ncomp=nl3,
                    dtype=ctypes.c_int32
                )
            elif self._escape_list.ncomp < nl3:
                self._escape_list.realloc(nl3)

            # --- find escapees ---

            nl  = self.state.get_position_dat().npart_local
            cuda_runtime.cuda_err_check(
                self._escape_guard_lib['cudaNProcPBCStageOne'](
                    ctypes.c_int32(nl),
                    self.state.domain.boundary.ctypes_data,
                    self.state.get_position_dat().ctypes_data,
                    self.state.domain.get_shift().ctypes_data,
                    self._escape_count.ctypes_data,
                    self._escape_dir_count.ctypes_data,
                    self._escape_list.ctypes_data
                )
            )

            dir_max = np.max(self._escape_dir_count[:])
            if self._escape_matrix is None:
                self._escape_matrix = host.Matrix(nrow=26,
                                                  ncol=dir_max,
                                                  dtype=ctypes.c_int32)

            elif self._escape_matrix.ncol < dir_max:
                self._escape_matrix.realloc(nrow=26, ncol=dir_max)


            # --- Populate escape matrix (essentially sort by direction)
            cuda_runtime.cuda_err_check(
                self._escape_guard_lib['cudaNProcPBCStageTwo'](
                    ctypes.c_int32(self._escape_count[0]),
                    ctypes.c_int32(self._escape_matrix.ncol),
                    self._escape_list.ctypes_data,
                    self._escape_matrix.ctypes_data
                )
            )


            # TODO pass matrix and dir counts to a cuda_state

















