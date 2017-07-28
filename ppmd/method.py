from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


_GRAPHICS = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _GRAPHICS = False

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    _GRAPHICS = False

# system level
import collections
import ctypes
import os
import re
import datetime
import inspect
import cProfile
import time
import math
import numpy as np

# package level
from ppmd import kernel, data, runtime, pio, mpi, opt, access, pairloop, loop

np.set_printoptions(threshold='nan')

_MPI = mpi.MPI
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier


###############################################################################
# Cell to Particle map handler for MD integrators
###############################################################################
class ListUpdateController(object):
    """
    The framework does not assume that it is employed in a MD simulation
    situation. This class implements cell list updating at a specific number of
    time steps or after a maximum velocity forces an update.
    """

    def __init__(self,
                 state_in=None,
                 step_count=1,
                 velocity_dat=None,
                 timestep=None,
                 shell_thickness=0.0,
                 step_index_func=None):

        self._state = state_in
        self._step_count = step_count
        self._velocity_dat = velocity_dat
        self._dt = timestep
        self._delta = shell_thickness

        self._moved_distance = 0.0
        self._test_count = 0
        self._step_counter = 0
        self._step_index = -10

        self._step_index_func = step_index_func

        self.boundary_method_timer = opt.Timer()
        self.check_status_timer = opt.Timer()

    def set_timestep(self, val):
        self._dt = val


    def increment_step_count(self):
        self._step_counter += 1

    def _get_max_moved_distance(self):
        """
        Get the maxium distance moved by a particle.
        :return:
        """
        if self._velocity_dat.npart_local > 0:
            return self._dt * self._velocity_dat.norm_linf()
        else:
            return 0.0

    def pre_update(self):
        """
        called after it is determined that an update is happening
        :return:
        """
        self.execute_boundary_conditions()

    def post_update(self):
        self._state.rebuild_cell_to_particle_maps()
        self._reset_moved_distance()


    def determine_update_status(self):
        """
        Return true if update of cell list is needed.
        :return:
        """
        self.check_status_timer.start()

        if self._step_index_func is not None:
            tmp = self._step_index_func()
            if tmp == self._step_index:
                #print "no update needed"
                self.check_status_timer.pause()
                opt.PROFILE[
                    self.__class__.__name__+':determine_update_status'
                ] = (self.check_status_timer.time())
                return False
            else:
                #print "update possibly needed"
                self._step_index = tmp


        self._test_count += 1
        self._moved_distance += self._get_max_moved_distance()

        if self._moved_distance >= 0.5 * self._delta:
            print("RANK %(RK)s  WARNING: Max velocity triggered list rebuild |" % \
                  {'RK':_MPIRANK}, _MPIRANK, "distance",\
                  self._moved_distance, "times reused", self._test_count, \
                  "dist:", 0.5 * self._delta)

        _ret = 0


        # print self._test_count, self._state.invalidate_lists, self._moved_distance
        if (self._moved_distance >= 0.5 * self._delta) or \
                (self._step_counter % self._step_count == 0) or \
                self._state.invalidate_lists:

            _ret = 1

        _ret_old = _ret

        _tmp = np.array([_ret], dtype=ctypes.c_int)
        _tmpr = np.array([-1], dtype=ctypes.c_int)
        _MPI.COMM_WORLD.Allreduce(_tmp, _tmpr, op=_MPI.LOR)
        _ret = _tmpr[0]

        if _ret_old == 1 and _ret != 1:
            print("update status reductypes.on error, rank:", _MPIRANK)

        # print "_ret", _ret, self._delta, self._step_counter, self._step_count
        self.check_status_timer.pause()
        opt.PROFILE[
            self.__class__.__name__+':determine_update_status'
        ] = (self.check_status_timer.time())

        return bool(_ret)

    def _reset_moved_distance(self):
        self._test_count = 0
        self._moved_distance = 0.0
        self._state.invalidate_lists = False


    def execute_boundary_conditions(self):
        """
        Execute the boundary conditions for the simulation.
        """

        if self._state.domain.boundary_condition is not None:

            self.boundary_method_timer.start()


            flag = self._state.domain.boundary_condition.apply()

            if flag > 0:
                # print "invalidating lists on BCs"
                self._state.invalidate_lists = True


            self.boundary_method_timer.pause()
            opt.PROFILE[
                self.__class__.__name__+':execute_boundary_conditions'
            ] = (self.boundary_method_timer.time())

        else:
            print("WARNING NO BOUNDARY CONDITION TO APPLY")

###############################################################################
# New Velocity Verlet Method
###############################################################################

class IntegratorRange(object):
    def __init__ (self,
            n,
            dt,
            velocities,
            list_reuse_count=1,
            list_reuse_distance=0.1,
            verbose=True,
            cprofile_dump=None
            ):
        self.verbose = verbose
        self._g = velocities.group
        self._update_controller = ListUpdateController(
            self._g,
            step_count=int(list_reuse_count),
            velocity_dat=velocities,
            timestep=float(dt),
            shell_thickness=float(list_reuse_distance),
            step_index_func=self._get_loop_index
        )

        self._setup_tracking()

        self._ix = -1
        self._nm1 = n-1

        self.timer = opt.SynchronizedTimer()

        if cprofile_dump is not None:
            self._cprof_dump = os.path.abspath(
                os.path.join(
                    os.getcwd(),
                    cprofile_dump
                )
            )
            self._pr = cProfile.Profile()
        else:
            self._cprof_dump = None
            self._pr = None

    def _setup_tracking(self):
        _suc = self._update_controller
        self._g.pre_update_funcs.append(
            _suc.pre_update
        )
        self._g.determine_update_funcs.append(
            _suc.determine_update_status
        )
        self._g.post_update_funcs.append(
           _suc.post_update
        )


    def _get_loop_index(self):
        return self._ix


    def __iter__(self):
        if self._pr is not None:
            self._pr.enable()
        self.timer.start()
        return self
    def __next__(self):
        return self.next()
    def next(self):
        if self._ix < self._nm1:
            self._update_controller.increment_step_count()
            self._ix += 1
            return self._ix
        else:
            self._g.get_cell_to_particle_map().reset_callbacks()
            self.timer.pause()
            if self._pr is not None:
                self._pr.disable()
                self._pr.dump_stats(
                    self._cprof_dump + '.' + str(
                        _MPIRANK
                    )
                )

            if self.verbose:
                if _MPIRANK == 0:
                    print(60*'=')
                tt = self.timer.stop(str='Integration time:')
                opt.PROFILE[
                    self.__class__.__name__+':loop_time'
                ] = tt


                if _MPIRANK == 0:
                    print(60*'-')
                    opt.print_profile()
                    print(60*'=')

            raise StopIteration
