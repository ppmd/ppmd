from __future__ import print_function, division, absolute_import
"""
CUDA implementations of methods to handle the cell decomposition of a domain.
"""

#system
import ctypes
import math
import numpy as np

#package
import ppmd.cuda
import ppmd.cuda.cuda_config
import ppmd.opt
import ppmd.opt as opt
import ppmd.host as host

#cuda
from ppmd.cuda import cuda_runtime, cuda_base, cuda_build


class SubCellOccupancyMatrix(object):
    def __init__(self, domain, cell_width, positions):

        self.domain = domain
        boundary = domain.boundary

        assert cell_width > 0, "bad cell width"
        assert boundary[1] > boundary[0], "nonsensical boundary"
        assert boundary[3] > boundary[2], "nonsensical boundary"
        assert boundary[5] > boundary[4], "nonsensical boundary"

        self.positions = positions

        self.cell_array = host.Array(ncomp=3, dtype=ctypes.c_int)
        self.cell_sizes = host.Array(ncomp=3, dtype=ctypes.c_double)

        # get sizes just considering interior
        cell_array = [0,0,0]
        cell_array[0] = int(float(boundary[1] - boundary[0])/cell_width)
        cell_array[1] = int(float(boundary[3] - boundary[2])/cell_width)
        cell_array[2] = int(float(boundary[5] - boundary[4])/cell_width)

        cell_sizes = [0,0,0]
        cell_sizes[0] = float(boundary[1] - boundary[0])/cell_array[0]
        cell_sizes[1] = float(boundary[3] - boundary[2])/cell_array[1]
        cell_sizes[2] = float(boundary[5] - boundary[4])/cell_array[2]
        self.cell_sizes[:] = cell_sizes[:]

        padx = int(math.ceil(self.domain.cell_edge_lengths[0]/cell_sizes[0])) +1
        pady = int(math.ceil(self.domain.cell_edge_lengths[1]/cell_sizes[1])) +1
        padz = int(math.ceil(self.domain.cell_edge_lengths[2]/cell_sizes[2])) +1

        rpadx = padx*cell_sizes[0]
        rpady = pady*cell_sizes[1]
        rpadz = padz*cell_sizes[2]

        #print "CA", cell_array[:], "CS", self.cell_sizes[:], "CES", self.domain.cell_edge_lengths[:]

        self.cell_array[0] = cell_array[0] + 2*padx
        self.cell_array[1] = cell_array[1] + 2*pady
        self.cell_array[2] = cell_array[2] + 2*padz

        #print "CA2", self.cell_array[:]



        self.boundary = host.Array(ncomp=6, dtype=ctypes.c_double)
        self.boundary[0] = boundary[0] - rpadx
        self.boundary[1] = boundary[1] + rpadx
        self.boundary[2] = boundary[2] - rpady
        self.boundary[3] = boundary[3] + rpady
        self.boundary[4] = boundary[4] - rpadz
        self.boundary[5] = boundary[5] + rpadz


        self.cell_count = cell_array[0]*cell_array[1]*cell_array[2]
        self.particle_layers = cuda_base.Array(ncomp=1, dtype=ctypes.c_int)
        self.cell_reverse_lookup = cuda_base.Array(ncomp=1, dtype=ctypes.c_int)
        self.cell_contents_count = cuda_base.Array(ncomp=self.cell_count, dtype=ctypes.c_int)
        self.matrix = cuda_base.Matrix(nrow=self.cell_count,
                                       ncol=1,
                                       dtype=ctypes.c_int)
        self.num_layers = 0

        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaSubCellOccupancyMatrixSource.cu', 'r') as fh:
            _code = fh.read()
        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaSubCellOccupancyMatrixSource.h', 'r') as fh:
            _header = fh.read()
        _name = 'SubCellOccupancyMatrix'
        lib = cuda_build.simple_lib_creator(_header, _code, _name)
        self._sort_lib = lib['LayerSort']
        self._fill_lib = lib['PopMatrix']

        self.version_id = 0

    def sort(self):

        assert self.positions.group is not None

        oldm = self.positions.group.get_cell_to_particle_map()
        old_num_layers = oldm.layers_per_cell
        old_num_cells = oldm.domain.cell_count
        old_count = old_num_layers*old_num_cells

        npart = self.positions.npart_local + self.positions.npart_local_halo


        _tpb = 256
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(old_count / float(_tpb))), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)

        ca = self.cell_array
        ca = ca[0] * ca[1] * ca[2]
        _blocksize2 = (ctypes.c_int * 3)(int(math.ceil(ca / float(_tpb))), 1, 1)
        _threadsize2 = (ctypes.c_int * 3)(_tpb, 1, 1)

        num_layers = ctypes.c_int(0)


        if self.particle_layers.ncomp < npart:
            self.particle_layers.realloc(ncomp=npart + 8)
        if self.cell_reverse_lookup.ncomp < npart:
            self.cell_reverse_lookup.realloc(ncomp=npart + 8)
        if self.cell_contents_count.ncomp < ca:
            self.cell_contents_count.realloc(ncomp=ca)

        self.cell_contents_count.zero()


        self._sort_lib(
            _blocksize, _threadsize,
            _blocksize2, _threadsize2,
            ctypes.c_int(old_count),
            ctypes.c_int(ca),
            ctypes.byref(num_layers),
            self.particle_layers.ctypes_data,
            self.cell_reverse_lookup.ctypes_data,
            self.cell_contents_count.ctypes_data,
            self.cell_array.ctypes_data,
            self.boundary.ctypes_data,
            self.cell_sizes.ctypes_data,
            self.positions.ctypes_data,
            ctypes.c_int(old_num_layers),
            oldm.matrix.ctypes_data,
            oldm.cell_contents_count.ctypes_data
        )

        self.num_layers = max(num_layers.value, self.num_layers)

        if self.matrix.nrow < ca or self.matrix.ncol < self.num_layers:
            self.matrix.realloc(nrow=ca, ncol=self.num_layers, copy=False)

        self._fill_lib(
            _blocksize, _threadsize,
            ctypes.c_int(old_count),
            ctypes.c_int(self.num_layers),
            self.particle_layers.ctypes_data,
            self.cell_reverse_lookup.ctypes_data,
            self.matrix.ctypes_data,
            ctypes.c_int(old_num_layers),
            oldm.matrix.ctypes_data,
            oldm.cell_contents_count.ctypes_data
        )

        #print self.cell_contents_count[:], self.num_layers




class CellOccupancyMatrix(object):
    """
    Class to compute and store a cell occupancy matrix for a domain and a set of positions
    """
    def __init__(self):
        self._init = False
        self._setup = False

        self.cell_contents_count = None
        """Number of particles per cell, determines number of layers per cell."""

        self.cell_reverse_lookup = None
        """Map between particle index and containing cell."""

        self.particle_layers = None
        """Stores which layer each particle is contained in."""

        self.matrix = None
        """The occupancy matrix."""

        # build vars
        self._p1_lib = None
        self._boundary = None
        self._cell_edge_lengths = None
        self._cell_array = None
        self.cell_in_halo_flag = None


        # setup vars
        self._n_func = None
        self._domain = None
        self._positions = None
        self._n_layers = 0


        self.update_required = True

        self._update_set = False
        self._update_func = None
        self._update_func_pre = None
        self._update_func_post = None

        self.version_id = 0
        self.version_id_halo = 0


        self._timer = ppmd.opt.Timer()

        # scan vars
        self._ccc_scan = cuda_base.Array(ncomp=1, dtype=ctypes.c_int)

    def cell_contents_count_scan(self):
        """
        Get the exclusive scan of the cell contents counts array
        :return:
        """

        self._ccc_scan.realloc_zeros(self._domain.cell_count)

        cuda_runtime.cuda_mem_cpy(
            self._ccc_scan.ctypes_data,
            self.cell_contents_count.ctypes_data,
            ctypes.c_size_t(
                self._domain.cell_count*ctypes.sizeof(
                    self.cell_contents_count.dtype
                )
            ),
            'cudaMemcpyDeviceToDevice'
        )

        cuda_runtime.LIB_CUDA_MISC['cudaExclusiveScanInt'](
            self._ccc_scan.ctypes_data,
            ctypes.c_int(self._ccc_scan.ncomp)
        )

        return self._ccc_scan


    def trigger_update(self):
        self.update_required = True


    def setup_update_tracking(self, func):
        """
        Setup an automatic cell update.
        :param func:
        """
        self._update_func = func
        self._update_set = True

    def setup_callback_on_update(self, func):
        """
        Setup a function to be ran after the cell list if updated.
        :param func: Function to run.
        :return:
        """
        self._update_func_post = func

    def setup_pre_update(self, func):
        self._update_func_pre = func

    def reset_callbacks(self):
        self._update_func = None
        self._update_func_pre = None
        self._update_func_post = None


    def _update_tracking(self):

        if self._update_func is None:
            return True

        if self._update_set and self._update_func():
            return True
        else:
            return False

    def _pre_update(self):
        """
        Run a pre update function eg boundary conditions.
        """
        if self._update_func_pre is not None:
            self._update_func_pre()
            # pass

    def create(self):
        self._cell_sort_setup()


    def check(self):
        """
        Check if the cell_linked_list needs updating and update if required.
        :return:
        """

        if not self._init:
            self._cell_sort_setup()

            if not self._init:
                return False


        if (self.update_required is True) or self._update_tracking():

            self._pre_update()

            self.sort()
            if self._update_func_post is not None:
                self._update_func_post()



                return True
        else:
            return False


    def setup(self, n_func=None, positions_in=None, domain_in=None):
        """
        Setup the cell occupancy matrix class
        :param n_func:
        :param positions_in:
        :param domain_in:
        :return:
        """
        assert n_func is not None, "No n_func passed."
        assert positions_in is not None, "No positions passed"
        assert domain_in is not None, "No domain passed"

        self._n_func = n_func
        self._domain = domain_in
        self._positions = positions_in

    def _cell_sort_setup(self):

        self.particle_layers = cuda_base.Array(ncomp=self._n_func(), dtype=ctypes.c_int)
        self.cell_reverse_lookup = cuda_base.Array(ncomp=self._n_func(), dtype=ctypes.c_int)
        self.cell_contents_count = cuda_base.Array(ncomp=self._domain.cell_count, dtype=ctypes.c_int)
        self.matrix = cuda_base.device_buffer_2d(nrow=self._domain.cell_count,
                                                 ncol=self._n_func()//self._domain.cell_count,
                                                 dtype=ctypes.c_int)


        self._n_layers = self.matrix.ncol
        self._n_cells = self.matrix.nrow
        

        #self._boundary = cuda_base.Array(initial_value=self._domain.boundary_outer)
        #self._cell_edge_lengths = cuda_base.Array(initial_value=self._domain.cell_edge_lengths)
        #self._cell_array = cuda_base.Array(initial_value=self._domain.cell_array, dtype=ctypes.c_int)



        self._setup = True
        self._build()


    def _build(self):
        """
        Build the library to create the cell occupancy matrix.
        :return:
        """
        assert self._setup is not False, "Run CellOccupancyMatrix.setup() first."

        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaCellOccupancyMatrixSource.cu', 'r') as fh:
            _code = fh.read()
        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaCellOccupancyMatrixSource.h', 'r') as fh:
            _header = fh.read()
        _name = 'CellOccupancyMatrix'

        self._p1_lib = cuda_build.simple_lib_creator(
            _header, _code, 'CellOccupancyMatrix')

        self._init = True

    def sort(self):

        self._timer.start()

        # Things that need to vary in size.
        if self.particle_layers.ncomp < self._n_func():
            self.particle_layers.realloc(self._n_func())
            self.cell_reverse_lookup.realloc(ncomp=self._n_func())

        if self.cell_contents_count.ncomp < self._domain.cell_count:
            self.cell_contents_count.realloc(self._domain.cell_count)
            self.matrix.realloc(nrow=self._domain.cell_count,
                                ncol=self.matrix.ncol)

        self._update_cell_in_halo()

        self.cell_contents_count.zero()

        _tpb = 512
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(self._n_func() / float(_tpb))), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)

        _tpb2 = 128
        
        _ca = self._domain.cell_array
        _ca = _ca[0] * _ca[1] * _ca[2]

        _blocksize2 = (ctypes.c_int * 3)(int(math.ceil(_ca / float(_tpb))), 1, 1)
        _threadsize2 = (ctypes.c_int * 3)(_tpb, 1, 1)
        
        _nl = ctypes.c_int(self._n_layers)
        _n_cells = ctypes.c_int(self._n_cells)

        args = [ctypes.c_int32(self._domain.comm.py2f()),
                ctypes.c_int(1),
                _blocksize,
                _threadsize,
                _blocksize2,
                _threadsize2,
                ctypes.c_int(self._n_func()),
                ctypes.c_int(_ca),
                ctypes.byref(_nl),
                ctypes.byref(_n_cells),
                self.particle_layers.ctypes_data,
                self.cell_reverse_lookup.ctypes_data,
                self.cell_contents_count.ctypes_data,
                ctypes.byref(self.matrix.ctypes_data),
                self._domain.cell_array.ctypes_data,
                self._domain.boundary.ctypes_data,
                self._domain.cell_edge_lengths.ctypes_data,
                self._positions.ctypes_data
                ]

        rval = self._p1_lib['LayerSort'](*args)


        self._n_layers = _nl.value
        self.matrix.ncol = self._n_layers

        self.version_id += 1
        self.update_required = False

        self._timer.pause()
        opt.PROFILE[
            self.__class__.__name__+':sort'
        ] = (self._timer.time())

    @property
    def layers_per_cell(self):
        return self._n_layers

    @property
    def domain(self):
        return self._domain

    @property
    def positions(self):
        return self._positions

    def prepare_halo_sort(self, max_halo_layers=None):
        assert max_halo_layers is not None, "no size passed"

        # Is a resize needed?
        if max_halo_layers > self._n_layers:
            print("resizing occupancy matrix, you should not see this message")

            new_matrix = cuda_base.device_buffer_2d(nrow=self.matrix.nrow,
                                                    ncol=max_halo_layers,
                                                    dtype=ctypes.c_int32)

            cuda_runtime.cuda_err_check(
            self._p1_lib['copy_matrix_cols'](
                                             ctypes.c_int32(self.matrix.ncol),
                                             ctypes.c_int32(new_matrix.ncol),
                                             ctypes.c_int32(new_matrix.nrow),
                                             self.matrix.ctypes_data,
                                             new_matrix.ctypes_data
                                            ))

            self.matrix.free()
            self.matrix = new_matrix


            self._n_layers = max_halo_layers

    def _update_cell_in_halo(self):


        if self._cell_array is None or \
            self._cell_array[0] != self._domain.cell_array[0] or \
            self._cell_array[1] != self._domain.cell_array[1] or \
            self._cell_array[2] != self._domain.cell_array[2]:
            # --
            self._cell_array = np.array(self._domain.cell_array[:])

            tl = self._cell_array[0] * self._cell_array[1] * self._cell_array[2]
            ca = self._cell_array
            tmp = np.ones((ca[2], ca[1], ca[0]), dtype=ctypes.c_int)

            tmp[1:ca[2]-1:,1:ca[1]-1:,1:ca[0]-1:] = 0
            tmp[2:ca[2]-2:,2:ca[1]-2:,2:ca[0]-2:] = -1

            tmp = tmp.ravel()

            self.cell_in_halo_flag = cuda_base.Array(ncomp=tl, dtype=ctypes.c_int)

            self.cell_in_halo_flag[:] = tmp





# Default
OCCUPANCY_MATRIX = None

class NeighbourListLayerBased(object):

    def __init__(self, occ_matrix=OCCUPANCY_MATRIX, cutoff=None):

        self._occ_matrix = occ_matrix
        assert cutoff is not None, "cuda_cell::NeighbourListLayerBased.setup error: No cutoff passed."
        self._rc = cutoff

        self.max_neigbours_per_particle = None
        self.version_id = 0

        self.list = cuda_base.Matrix(nrow=1, ncol=1, dtype=ctypes.c_int)

        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaNeighbourListGenericSource.cu', 'r') as fh:
            _code = fh.read()
        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaNeighbourListGenericSource.h', 'r') as fh:
            _header = fh.read()
        _name = 'NeighbourList'
        self._lib = cuda_build.simple_lib_creator(_header, _code, _name)[_name]

        self._timer = ppmd.opt.Timer()


    def update(self):
        self._timer.start()

        limit = 8 * self._occ_matrix.layers_per_cell + 1

        if (self.list.ncol < self._occ_matrix.positions.npart_local) or \
                (self.list.nrow < limit):

            self.list.realloc(
                nrow=limit,
                ncol=max(
                    int(self._occ_matrix.positions.npart_local)
                    ,1
                ),
                copy=False
            )

        _tpb = 256
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(
            self._occ_matrix.positions.npart_local / float(_tpb)
        )), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)


        args = (
            _blocksize,
            _threadsize,
            ctypes.c_int(limit), # nmax
            ctypes.c_int(self._occ_matrix.positions.npart_local),      # npart
            ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
            ctypes.c_double(self._rc ** 2),                      # cutoff squared
            self._occ_matrix.domain.cell_array.ctypes_data,
            self._occ_matrix.positions.struct,
            self._occ_matrix.cell_reverse_lookup.struct,
            self._occ_matrix.matrix.struct,
            self._occ_matrix.cell_contents_count.struct,
            self.list.struct
        )

        e = self._lib(*args)

        if e < 0:
            print("warning reallocing neighbour matrix")
            limit = -1*e + 1
            self.list.realloc(nrow=limit, ncol=self._occ_matrix.positions.npart_local)

            args = (
                _blocksize,
                _threadsize,
                ctypes.c_int(limit), # nmax
                ctypes.c_int(self._occ_matrix.positions.npart_local),      # npart
                ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
                ctypes.c_double(self._rc ** 2),                      # cutoff squared
                self._occ_matrix.domain.cell_array.ctypes_data,
                self._occ_matrix.positions.struct,
                self._occ_matrix.cell_reverse_lookup.struct,
                self._occ_matrix.matrix.struct,
                self._occ_matrix.cell_contents_count.struct,
                self.list.struct
            )

            e = self._lib(*args)

        if e < 0:
            raise Exception

        self.max_neigbours_per_particle = limit
        self.version_id = self._occ_matrix.version_id

        self._timer.pause()
        opt.PROFILE[
            self.__class__.__name__+':update'
        ] = (self._timer.time())

class NeighbourListLayerSplit(object):

    def __init__(self, occ_matrix=OCCUPANCY_MATRIX, cutoff=None):

        self._occ_matrix = occ_matrix
        assert cutoff is not None, "cuda_cell::NeighbourListLayerBased.setup error: No cutoff passed."
        self._rc = cutoff

        self.max_neigbours_per_particle = None
        self.version_id_1 = 0
        self.version_id_2 = 0

        self.list1 = cuda_base.Matrix(nrow=1, ncol=1, dtype=ctypes.c_int)
        self.list2 = cuda_base.Matrix(nrow=1, ncol=1, dtype=ctypes.c_int)

        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaNeighbourListSplitSource.cu', 'r') as fh:
            _code = fh.read()
        with open(str(
                ppmd.cuda.cuda_config.LIB_DIR) + '/cudaNeighbourListSplitSource.h', 'r') as fh:
            _header = fh.read()
        _name1 = 'NeighbourList'
        _name2 = 'NeighbourList2'
        _lib = cuda_build.simple_lib_creator(_header, _code, _name1)
        self._lib1 = _lib[_name1]
        self._lib2 = _lib[_name2]


    def update1(self):

        limit = 8 * self._occ_matrix.layers_per_cell + 1

        if (self.list1.ncol < self._occ_matrix.positions.npart_local) or \
                (self.list1.nrow < limit):
            self.list1.realloc(nrow=limit,
                               ncol=self._occ_matrix.positions.npart_local)

        _tpb = 256
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(
            self._occ_matrix.positions.npart_local / float(_tpb)
        )), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)


        args = (
            _blocksize,
            _threadsize,
            ctypes.c_int(limit), # nmax
            ctypes.c_int(self._occ_matrix.positions.npart_local),# npart
            ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
            ctypes.c_double(self._rc ** 2),                      # cutoff squared
            self._occ_matrix.domain.cell_array.ctypes_data,
            self._occ_matrix.cell_in_halo_flag.struct,
            self._occ_matrix.positions.struct,
            self._occ_matrix.cell_reverse_lookup.struct,
            self._occ_matrix.matrix.struct,
            self._occ_matrix.cell_contents_count.struct,
            self.list1.struct
        )

        e = self._lib1(*args)

        if e < 0:
            print("warning reallocing neighbour matrix")
            limit = -1*e + 1
            self.list1.realloc(nrow=limit, ncol=self._occ_matrix.positions.npart_local)

            args = (
                _blocksize,
                _threadsize,
                ctypes.c_int(limit), # nmax
                ctypes.c_int(self._occ_matrix.positions.npart_local),      # npart
                ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
                ctypes.c_double(self._rc ** 2),                      # cutoff squared
                self._occ_matrix.domain.cell_array.ctypes_data,
                self._occ_matrix.positions.struct,
                self._occ_matrix.cell_reverse_lookup.struct,
                self._occ_matrix.matrix.struct,
                self._occ_matrix.cell_contents_count.struct,
                self.list1.struct
            )

            e = self._lib1(*args)

        if e < 0:
            raise Exception

        self.max_neigbours_per_particle = limit

        self.version_id_1 = self._occ_matrix.version_id


    def update2(self):

        limit = 8 * self._occ_matrix.layers_per_cell + 1

        if (self.list2.ncol < self._occ_matrix.positions.npart_local) or \
            (self.list2.nrow < limit):
            self.list2.realloc(nrow=limit,
                               ncol=self._occ_matrix.positions.npart_local)

        _tpb = 256
        _blocksize = (ctypes.c_int * 3)(int(math.ceil(
            self._occ_matrix.positions.npart_local / float(_tpb)
        )), 1, 1)
        _threadsize = (ctypes.c_int * 3)(_tpb, 1, 1)


        args = (
            _blocksize,
            _threadsize,
            ctypes.c_int(limit), # nmax
            ctypes.c_int(self._occ_matrix.positions.npart_local),# npart
            ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
            ctypes.c_double(self._rc ** 2),                      # cutoff squared
            self._occ_matrix.domain.cell_array.ctypes_data,
            self._occ_matrix.cell_in_halo_flag.struct,
            self._occ_matrix.positions.struct,
            self._occ_matrix.cell_reverse_lookup.struct,
            self._occ_matrix.matrix.struct,
            self._occ_matrix.cell_contents_count.struct,
            self.list2.struct
        )

        e = self._lib2(*args)

        if e < 0:
            print("warning reallocing neighbour matrix")
            limit = -1*e + 1
            self.list2.realloc(nrow=limit, ncol=self._occ_matrix.positions.npart_local)

            args = (
                _blocksize,
                _threadsize,
                ctypes.c_int(limit), # nmax
                ctypes.c_int(self._occ_matrix.positions.npart_local),      # npart
                ctypes.c_int(self._occ_matrix.layers_per_cell),      # nlayers max
                ctypes.c_double(self._rc ** 2),                      # cutoff squared
                self._occ_matrix.domain.cell_array.ctypes_data,
                self._occ_matrix.positions.struct,
                self._occ_matrix.cell_reverse_lookup.struct,
                self._occ_matrix.matrix.struct,
                self._occ_matrix.cell_contents_count.struct,
                self.list2.struct
            )

            e = self._lib2(*args)

        if e < 0:
            raise Exception
        self.max_neigbours_per_particle = limit

        self.version_id_2 = self._occ_matrix.version_id























