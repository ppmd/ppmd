# system level
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# package level
from ppmd import data

class StateHandler(object):

    def __init__(self, state, shell_cutoff, pair=True, threaded=True):

        self.shell_cutoff = shell_cutoff
        self._group = state
        self._pair = pair
        self._threaded = threaded

        if self._group is not None:
            flag = self._group.cell_decompose(self.shell_cutoff)

    def pre_execute(self, dats):
        """
        The halo exchange process may reallocate all in the group. Hence this
        function loops over all dats to ensure pointers collected in the second
        pass over the dats are valid.
        """
        _group = self._group
        if _group is None:
            for pd in dats.values():
                if issubclass(type(pd[0]), data.PositionDat):
                    pd[0].group.cell_decompose(self.shell_cutoff)
                    _group = pd[0].group
                    break
        if _group is None:
            raise RuntimeError("No state/group found")

        cell2part = _group.get_cell_to_particle_map()
        cell2part.check()

        for d in dats.values():
            if type(d) is not tuple:
                print(d)
                raise RuntimeError("dat was not tuple")
            d[0].ctypes_data_access(d[1], pair=self._pair)

        return _group.npart_local, _group.npart_halo, \
            _group.get_cell_to_particle_map().max_cell_contents_count

    def get_pointer(self, dat):
        obj = dat[0]
        mode = dat[1]
        if issubclass(type(obj), data.GlobalArrayClassic):
            return obj.ctypes_data_access(mode, pair=self._pair, threaded=True)
        else:
            return obj.ctypes_data_access(mode, pair=self._pair)

    def post_execute(self, dats):
        for d in dats.values():
            assert type(d) is tuple
            obj = d[0]
            mode = d[1]
            if issubclass(type(obj), data.GlobalArrayClassic):
                obj.ctypes_data_post(mode, threaded=self._threaded)
            else:
                obj.ctypes_data_post(mode)



