from __future__ import print_function, division, absolute_import
import itertools
import ctypes
import numpy as np

from mpi4py import MPI
from ppmd import pygcl

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"


class cube_owner_map(object):
    """
    Compute the global map from cube index (lexicographic) to owning MPI rank
    and map to MPI ranks with sub-domains of non-empty intersection with each
    cube.
    
    Assumes that in each dimension ranks are assigned equal shares of the 
    domain.
    
    :param cart_comm: MPI cartesian communicator.
    :param cube_side_count: Integer subdivision level of domain.
    :param group_children: Consider cube subdivision to be the final mesh l of
    an octal tree and group cubes such that all children of cubes in mesh l-1
    are on the same MPI rank.
    """
    def __init__(self, cart_comm, cube_side_count, group_children=False):

        self.cart_comm = cart_comm
        """Cartesian communicator used"""
        topo = self.cart_comm.Get_topo()
        # dim order here is z, y, x
        dims = topo[0]
        top = topo[2]

        self.cube_count = cube_side_count**len(dims)
        """Total number of cubes"""
        self.cube_side_count = cube_side_count
        """Side count of octal tree on domain"""

        if group_children and not cube_side_count % 2 == 0:
            raise RuntimeError("Side length must be even with group_children")

        owners, contribs, lsize, loffset = self.compute_grid_ownership(
            dims, top, cube_side_count, group_children)

        cube_to_mpi = self.compute_map_product_owners(cart_comm, owners)
        starts, con_ranks, send_ranks = self.compute_map_product_contribs(
            cart_comm, cube_to_mpi, contribs)

        self.owners = owners

        self.cube_to_mpi = cube_to_mpi
        """Array from cube global index to owning mpi rank"""
        self.contrib_mpi = con_ranks
        """Array containing ranks contributing to a cube"""
        self.contrib_starts = starts
        """Cube i has contributing coefficients from ranks x_s_i to x_s_{i-1}
        where s is contrib_starts and x is contrib_mpi"""
        self.cube_to_send = send_ranks
        """Array if this mpi rank contributes to cube i then x_i is owning
        rank of cube i else x_i = -1"""
        self.local_size = np.array(lsize, dtype=ctypes.c_uint64)
        self.local_offset = np.array(loffset, dtype=ctypes.c_uint64)

    @staticmethod
    def compute_grid_ownership(dims, top, cube_side_count, group_children=False):
        """
        For each dimension compute the owner and contributing ranks. Does
        not perform out product to compute the full map.
        :param dims: tuple of mpi dims.
        :param top: tuple of top.
        :param cube_side_count: int, number of cubes in each dimension
        :param group_children: See class doc string.
        :return: tuple: ranks of each cube owner, contributing ranks.
        """

        ndim = len(dims)
        oocsc = 1.0 / cube_side_count
        hoocsc = 0.5 / cube_side_count

        if group_children:
            cube_mids = [oocsc*ix + ((ix + 1) % 2) * oocsc
                         for ix in range(cube_side_count)]
        else:
            cube_mids = [hoocsc + oocsc*ix for ix in range(cube_side_count)]

        def cube_lower_edge(x): return x * oocsc

        def cube_upper_edge(x): return (x + 1) * oocsc

        dim_owners = [[-1] * cube_side_count for nx in range(ndim)]

        # expanding empty lists with []*n gives [] not [[],...,[]]
        dim_contribs = [[[] for iy in range(cube_side_count)] for
                        ix in range(ndim)]

        loffset = [[-1] for nx in range(ndim)]
        lsize = [[-1] for nx in range(ndim)]

        for dx in range(ndim):

            def lbound(argx): return float(argx)/dims[dx]
            def ubound(argx): return float(argx+1)/dims[dx]

            inter = []

            for mx in range(dims[dx]):
                for cx, cmid in enumerate(cube_mids):
                    if (lbound(mx) < cmid) and (cmid <= ubound(mx)):
                        # this rank is the cube's unique owner
                        dim_owners[dx][cx] = mx

                    if (ubound(mx) > cube_lower_edge(cx)) and \
                            (lbound(mx) < cube_upper_edge(cx)):

                        # this rank's subdomain intersects the cube's lower
                        # bound
                        dim_contribs[dx][cx].append(mx)

                        # if the rank we are inspecting is this rank
                        if top[dx] == mx:
                            inter.append(cx)

                if top[dx] == mx:
                    loffset[dx] = inter[0]
                    lsize[dx] = len(inter)

        # dim owner orders is z dim then y dim ...
        return dim_owners, dim_contribs, lsize, loffset

    @staticmethod
    def compute_map_product_owners(cart_comm, dim_owners):
        """
        Compute the full map from cube index to mpi rank owner.
        :param cart_comm: MPI cart_comm to use for logical coord to rank 
        conversion.
        :param dim_owners: ndims x cube_side_count tuple representing cube to 
        owner map per dimension, elements are ints.
        lists of contributors.
        :return: array, length number of cubes, cube global index to mpi rank
        owner.
        """
        # domain is a cube therefore we can inspect one dimension
        ncubes_per_side = len(dim_owners[0])
        ndim = len(dim_owners)
        ncubes = ncubes_per_side**ndim
        iterset = [range(ncubes_per_side)] * ndim
        cube_to_mpi = np.zeros(ncubes, dtype=ctypes.c_uint64)

        tuple_to_lin_coeff = [
            ncubes_per_side**(ndim-dx-1) for dx in range(ndim)]

        # should convert an z,y,x tuple to lexicographic linear index
        def cube_tuple_to_lin(xarg): return sum([k[0]*k[1] for k in zip(
            xarg, tuple_to_lin_coeff)])

        # loop over cube indexes
        for cube_tuple in itertools.product(*iterset):
            cx = cube_tuple_to_lin(cube_tuple)
            ompi = cart_comm.Get_cart_rank([
                dim_owners[i][cube_tuple[i]] for i in range(ndim)])
            cube_to_mpi[cx] = ompi

        return cube_to_mpi

    @staticmethod
    def compute_map_product_contribs(cart_comm, cube_to_mpi, dim_contribs):
        """
        Compute the full map from cube index to mpi rank owner, cube index to
        mpi ranks of contributors.
        :param cart_comm: MPI cart_comm to use for logical coord to rank 
        conversion. Owner map per dimension, elements are ints.
        :param cube_to_mpi: ndims x cube_side_count tuple representing cube to 
        owner map per dimension, elements are ints.        
        :param dim_contribs: tuple ndims x cube_side_count tuple containing
        lists of contributors.
        :return: 
        """
        ncubes_per_side = len(dim_contribs[0])
        ndim = len(dim_contribs)
        my_rank = cart_comm.Get_rank()

        # duplicate function (makes testing easier)
        tuple_to_lin_coeff = [
            ncubes_per_side**(ndim-dx-1) for dx in range(ndim)]

        # should convert an z,y,x tuple to lexicographic linear index
        def cube_tuple_to_lin(xarg): return sum([k[0]*k[1] for k in zip(
            xarg, tuple_to_lin_coeff)])

        # loop over the cubes, then loop over contributors and add to contrib
        # list if contributing mpi rank is not owning rank.

        curr_start = 0
        starts = np.zeros(ncubes_per_side**ndim + 1, dtype=ctypes.c_uint64)
        con_ranks = list()
        send_ranks = np.zeros(ncubes_per_side**ndim, dtype=ctypes.c_int64)
        send_ranks[:] = -1

        iterset = [range(ncubes_per_side)] * ndim
        for ctx in itertools.product(*iterset):
            cx = cube_tuple_to_lin(ctx)
            owner = cube_to_mpi[cx]

            # loop over contributing ranks
            for con in itertools.product(
                    *[dim_contribs[i][j] for i, j in enumerate(ctx)]):
                con_rank = cart_comm.Get_cart_rank(con)

                # build list this rank sends to
                if con_rank != owner:
                    if con_rank == my_rank:
                        send_ranks[cx] = owner
                    elif my_rank == owner:
                        curr_start += 1
                        con_ranks.append(con_rank)

            starts[cx + 1] = curr_start

        starts = np.array(starts, dtype=ctypes.c_uint64)
        con_ranks = np.array(con_ranks, dtype=ctypes.c_uint64)

        return starts, con_ranks, send_ranks


def compute_local_size_offset(owners, dims, top):

    ndim = len(owners)
    sizes = [[] for dx in range(ndim)]
    offsets = [[] for dx in range(ndim)]
    for d in range(ndim):
        for rx in range(dims[d]):
            offsets[d].append(sum(sizes[d]))
            sizes[d].append(owners[d].count(rx))

    mysize = []
    myoffset = []
    for d in range(ndim):
        mysize.append(sizes[d][top[d]])
        myoffset.append(offsets[d][top[d]])

    return mysize, myoffset


def compute_interaction_offsets(cube_index):
    """
    Compute the interaction offsets for a given child cube. Child cubes are
    indexed with a tuple.
    :param cube_index: Tuple of child cube to compute offsets for.
    :return: numpy array type ctypes.c_int32 size 189x3 of offsets.
    """
    ox = np.array((cube_index[0], cube_index[1], cube_index[2]))
    if np.sum(ox**2) > 3 or not np.all(np.abs(ox) == ox):
        raise RuntimeError("unexpected cube_index" + str(cube_index))

    # get 6x6x6 array of all offsets for the child cube.
    ogrid = np.array([range(-2 - cube_index[0], 4 - cube_index[0]),
                      range(-2 - cube_index[1], 4 - cube_index[1]),
                      range(-2 - cube_index[2], 4 - cube_index[2])])

    # loop over all offsets, ignore the 27 nearest neighbours.
    ro = np.zeros(shape=(189, 3), dtype=ctypes.c_int32)
    ri = 0
    for iz in range(6):
        for iy in range(6):
            for ix in range(6):
                ox = np.array((ogrid[0, ix], ogrid[1, iy], ogrid[2, iz]))
                if not np.sum(ox**2) <= 3:
                    ro[ri, :] = ox
                    ri += 1

    if ri != 189:
        raise RuntimeError('Expected 189 cube offsets but recorded ' + str(ri))
    return ro


def compute_interaction_tlookup():
    """
    Compute the 8x189 lookup table to map from offset index to recomputed
    spherical harmonics. Theta part.
    """
    def tuple_to_lin(x): return (x[:, 2]+3)*49 + (x[:, 1]+3)*7 + x[:, 0]+3

    ro = np.zeros(shape=(8, 189), dtype=ctypes.c_int32)
    ri = 0
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                ro[ri, :] = tuple_to_lin(
                    compute_interaction_offsets((ix, iy, iz)))
                ri += 1
    return ro


def compute_interaction_plookup():
    """
    Compute the 8x189 lookup table to map from offset index to recomputed
    spherical harmonics. Phi part.
    """
    def tuple_to_lin(x): return (x[:, 1]+3)*7 + x[:, 0]+3

    ro = np.zeros(shape=(8, 189), dtype=ctypes.c_int32)
    ri = 0
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                ro[ri, :] = tuple_to_lin(
                    compute_interaction_offsets((ix, iy, iz)))
                ri += 1
    return ro


def compute_interaction_radius():
    """
    Compute the coefficients such that when multiplied by the cube with they 
    give the radius to the box centre.
    """
    def tuple_to_coeff(x): return (x[:,0]**2. + x[:,1]**2. + x[:,2]**2.)**0.5
    ro = np.zeros(shape=(8, 189), dtype=ctypes.c_double)
    ri = 0
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                ro[ri, :] = tuple_to_coeff(
                    compute_interaction_offsets((ix, iy, iz)))
                ri += 1
    return ro


def compute_interaction_lists(local_size):
    """
    Compute the local interaction offset lists for a local domain. Child cubes
    are indexed lexicographically from 0.
    :param local_size: tuple of local cube domain dimensions.
    :return: 8x189 ctypes.c_int32 offsets numpy array.
    """

    if not local_size[0] > 5 or not local_size[1] > 5 \
            or not local_size[2] > 5:
        raise RuntimeError(
            "local size too small in a dimension: " + str(local_size))

    def tuple_to_lin(x): return x[:, 2]*local_size[0]*local_size[1] + \
        x[:, 1]*local_size[0] + x[:, 0]

    ro = np.zeros(shape=(8, 189), dtype=ctypes.c_int32)
    ri = 0
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                ro[ri, :] = tuple_to_lin(
                    compute_interaction_offsets((ix, iy, iz)))
                ri += 1

    if ri != 8:
        raise RuntimeError("unexpected number of offset lists " + str(ri))

    return ro


def cube_tuple_to_lin_xyz(ii, n):
    """
    convert xyz tuple ii to linear index in cube of side length n.
    :param ii:  (x,y,z) tuple.
    :param n: side length
    """
    return int(ii[0] + n*(ii[1] + n*ii[2]))


def cube_tuple_to_lin_zyx(ii, n):
    """
    convert xyz tuple ii to linear index in cube of side length n.
    :param ii:  (z,y,x) tuple.
    :param n: side length
    """
    return int(ii[2] + n*(ii[1] + n*ii[0]))

class OctalGridLevel(object):
    def __init__(self, level, parent_comm, entry_map=None):
        """
        Level in the octal tree.  
        :param level: Non-negative integer subdivision level.
        :param parent_comm: Cartesian communicator to potentially split to 
        form the cartesian communicator for this level.
        """
        if level < 0:
            raise RuntimeError("Expected non-negative level integer.")

        self.level = level
        self.parent_comm = parent_comm
        self.ncubes_global = (2**level)**3
        self.ncubes_side_global = 2**level

        self.comm = parent_comm
        self.new_comm = False
        self.local_grid_cube_size = None
        """Size of grid"""
        self.parent_local_size = None
        """Size of parent grid with matching ownership"""
        self.local_grid_offset = None
        """Offset in global coodinates to the local sub-domain"""
        self.grid_cube_size = None
        """Size of grid plus halos"""
        self._halo_exchange_method = None
        self.owners = np.zeros(shape=(2**level, 2**level, 2**level),
                               dtype=ctypes.c_uint32)
        """Map from global cube index to owning MPI rank"""
        self.global_to_local = np.zeros(shape=(2**level, 2**level, 2**level),
                               dtype=ctypes.c_uint32)
        """Map from global cube index to local cube index"""
        self.global_to_local_halo = np.zeros(
            shape=(2**level, 2**level, 2**level), dtype=ctypes.c_uint32)
        """Map from global cube index to local cube index with halos"""
        self.global_to_local_parent = np.zeros(
            shape=(int(2**(level-1)), int(2**(level-1)), int(2**(level-1))),
            dtype=ctypes.c_uint32)
        """Map from global cube index to local parent index"""

        self.global_to_local_parent[:] = -1
        self.global_to_local[:] = -1

        if parent_comm != MPI.COMM_NULL:
            self._init_comm(parent_comm)
            self._init_decomp(parent_comm, entry_map)
        if self.comm != MPI.COMM_NULL:
            self._init_halo_exchange_method()

        self.nbytes = self.owners.nbytes + self.global_to_local.nbytes +\
            self.global_to_local_halo.nbytes

    def _init_comm(self, parent_comm):
        # set the min work per process at a 2x2x2 block of cubes for levels>1.
        work_units_per_side = 2 ** (self.level - 1) if self.level > 1 else 1

        parent_rank = parent_comm.Get_rank()

        current_dims = parent_comm.Get_topo()[0]
        new_dims = (min(current_dims[0], work_units_per_side),
                    min(current_dims[1], work_units_per_side),
                    min(current_dims[2], work_units_per_side))

        # is the parent communicator too large?
        if not(new_dims[0] == current_dims[0] and
                new_dims[1] == current_dims[1] and
                new_dims[2] == current_dims[2]):

            work_units = new_dims[0] * new_dims[1] * new_dims[2]

            color = 0 if parent_rank < work_units else MPI.UNDEFINED
            tmp_comm = parent_comm.Split(color=color, key=parent_rank)
            
            if tmp_comm != MPI.COMM_NULL:
                self.comm = tmp_comm.Create_cart(dims=new_dims,
                    periods=(1, 1, 1), reorder=False)
            else:
                self.comm = MPI.COMM_NULL
            self.new_comm = True


    def _init_decomp(self, parent_comm, entry_map=None):
        if self.comm != MPI.COMM_NULL:
            dims = self.comm.Get_topo()[0]
            top = self.comm.Get_topo()[2]
            work_units_per_side = 2 ** (self.level - 1) if self.level > 0 \
                                  else 1
            ndim = len(dims)

            if self.level > 0:
                if entry_map is None:
                    owners = [[] for nx in range(ndim)]
                    # need to compute owners
                    for d in range(ndim):
                        base_size = work_units_per_side // dims[d]
                        sizes = [base_size for nx in range(dims[d])]
                        for ex in range(work_units_per_side \
                                                - base_size*dims[d]):
                            sizes[ex] += 1

                        for nx in range(dims[d]):
                            for mx in range(sizes[nx]):
                                owners[d] += [nx, nx]

                else:
                    owners = entry_map.owners

                lt, lo = compute_local_size_offset(
                    owners, dims, top)

            else:
                lt = (1, 1, 1)
                lo = (0, 0, 0)
                owners = ([0], [0], [0])


            self.local_grid_cube_size = np.array(lt, dtype=ctypes.c_uint32)
            self.parent_local_size = np.array(
                (int(lt[0]//2), int(lt[1]//2), int(lt[2]//2)),
                dtype=ctypes.c_uint32)
            self.grid_cube_size = np.array((lt[0] + 4, lt[1] + 4, lt[2] + 4),
                                           dtype=ctypes.c_uint32)
            self.local_grid_offset = np.array(lo, dtype=ctypes.c_uint32)

            # compute the maps from global id to local id by looping over local
            # ids
            for ii in itertools.product(
                    range(lt[0]), range(lt[1]), range(lt[2])):
                gid_tuple = (ii[0] + lo[0], ii[1] + lo[1], ii[2] + lo[2])
                gid = cube_tuple_to_lin_zyx(gid_tuple,
                                            self.ncubes_side_global)

                self.global_to_local.ravel()[gid] = ii[2] + \
                                                    lt[2]*(ii[1] + lt[1]*ii[0])

                self.global_to_local_halo.ravel()[gid] = ii[2] + 2 +\
                                                 (lt[2] + 4)*(ii[1] + 2 +
                                                 (lt[1] + 4)*(ii[0] + 2))
            for ii in itertools.product(
                    range(lt[0]//2), range(lt[1]//2), range(lt[2]//2)):
                gid_tuple = (ii[0] + lo[0]//2,
                             ii[1] + lo[1]//2,
                             ii[2] + lo[2]//2)
                gid = cube_tuple_to_lin_zyx(gid_tuple,
                                            self.ncubes_side_global/2)
                self.global_to_local_parent.ravel()[gid] = ii[2] + \
                    lt[2]*(ii[1] + (lt[1]//2)*ii[0])//2

            # outer product
            if self.level > 0:

                wups2 = work_units_per_side * 2
                for iz in range(wups2):
                    for iy in range(wups2):
                        for ix in range(wups2):

                            self.owners[iz, iy, ix] = owners[2][ix] + \
                                                      dims[2] * (owners[1][iy] +
                                                      dims[1] * owners[0][iz])

            else:
                self.owners[0, 0, 0] = 0

        # need the owner map on the parent comm ranks to send data up and down
        # the tree
        color = MPI.UNDEFINED
        if self.comm != MPI.COMM_NULL:
            if self.comm.Get_rank() == 0: color = 0
        else:
            color = 0
        remain_comm = parent_comm.Split(color=color,
                                        key=parent_comm.Get_rank())
        if color != MPI.UNDEFINED:
            remain_comm.Bcast(self.owners[:], root=0)
            remain_comm.Free()


    def _init_halo_exchange_method(self):
        self._halo_exchange_method = pygcl.HaloExchange3D(self.comm,
                                                          self.grid_cube_size,
                                                          (2,2,2))

    def halo_exchange(self, arr):
        """
        Halo exchange the passed numpy array of shape self.grid_cube_size.
        :param arr: numpy array to exchange.
        """
        if self.comm != MPI.COMM_NULL:
            self._halo_exchange_method.exchange(arr)


class OctalTree(object):
    def __init__(self, num_levels, cart_comm):
        """
        Create an octal tree as a list of OctalLevels.
        :param num_levels: Number of levels in tree.
        :param cart_comm: comm to use at finest level.
        """
        self.num_levels = num_levels
        self.cart_comm = cart_comm
        self.levels = []
        comm_tmp = cart_comm

        self.entry_map = cube_owner_map(cart_comm, 2 ** (num_levels - 1), True)

        # work up tree from finest level as largest cart_comm is on the finest
        # level
        for lx in range(self.num_levels - 1, -1, -1):
            m = self.entry_map if lx == self.num_levels - 1 else None
            level_tmp = OctalGridLevel(level=lx, parent_comm=comm_tmp,
                                       entry_map=m)

            self.levels.append(level_tmp)
            comm_tmp = level_tmp.comm
        self.levels.reverse()

        self.nbytes = sum([lx.nbytes for lx in self.levels])

        if self.cart_comm is not self.levels[-1].comm:
            raise NotImplementedError(
                '''Finest level must use domain cart_comm, use more levels 
                or less MPI ranks.'''
            )


    def __getitem__(self, item):
        return self.levels[item]


class EntryData(object):
    def __init__(self, tree, ncomp, dtype=ctypes.c_double):
        self.tree = tree
        self.dtype = dtype
        self.ncomp = ncomp
        self.data = np.zeros(list(tree.entry_map.local_size) + [ncomp],
                             dtype=dtype)
        self.local_offset = tree.entry_map.local_offset
        self.local_size = tree.entry_map.local_size

        self._start = (self.local_offset[0],
                       self.local_offset[1],
                       self.local_offset[2])

        self._end = (self._start[0] + self.local_size[0],
                     self._start[1] + self.local_size[1],
                     self._start[2] + self.local_size[2])

    def add_onto(self, octal_data_tree):
        """
        add data onto a OctalDataTree of mode='halo'
        :param octal_data_tree: 
        """
        if octal_data_tree.tree is not self.tree:
            raise RuntimeError('cannot push data onto different tree')
        if octal_data_tree.ncomp != self.ncomp:
            raise RuntimeError('number of components mismatch')
        if octal_data_tree.dtype != self.dtype:
            raise RuntimeError('data type miss-match')

        ns = 2 ** (self.tree.num_levels - 1)
        comm = self.tree.cart_comm
        rank = comm.Get_rank()
        ncomp = self.ncomp
        entry_map = self.tree.entry_map

        dst_owners = self.tree[-1].owners.ravel()
        dst_g2l = self.tree[-1].global_to_local_halo

        dst_contribs = entry_map.contrib_mpi
        dst_contribs_starts = entry_map.contrib_starts
        dst_sends = entry_map.cube_to_send

        dst = octal_data_tree[-1].ravel()
        src = self.data.ravel()

        tmp = np.zeros(ncomp, dtype=self.dtype)

        send_req = None

        for cxt in itertools.product(range(ns), range(ns), range(ns)):
            cx = cube_tuple_to_lin_zyx(cxt, ns)

            local_ind = self._inside_entry(cxt)
            dst_owner = dst_owners[cx]

            if dst_owner == rank:
                dst_ind_b = dst_g2l[cxt]*ncomp
                dst_ind_e = dst_ind_b + ncomp

                # start by copying the local data
                src_ind_b = local_ind * ncomp
                src_ind_e = src_ind_b + ncomp
                dst[dst_ind_b: dst_ind_e:] = src[src_ind_b:src_ind_e:]

                # need to copy in data from surrounding contributors
                for nx in range(dst_contribs_starts[cx],
                                dst_contribs_starts[cx+1]):
                    comm.Recv(tmp[:], MPI.ANY_SOURCE, tag=cx)
                    dst[dst_ind_b: dst_ind_e:] += tmp[:]

            elif dst_sends[cx] > -1:
                # this rank needs to send it's contribution
                src_ind_b = local_ind * ncomp
                src_ind_e = src_ind_b + ncomp
                if send_req is not None: send_req.wait()
                send_req = comm.Isend(src[src_ind_b:src_ind_e:],
                                      dst_sends[cx], tag=cx)

        if send_req is not None: send_req.wait()

    def _inside_entry(self, p):
        if self._start[0] <= p[0] < self._end[0] and \
            self._start[1] <= p[1] < self._end[1] and \
            self._start[2] <= p[2] < self._end[2]:

            x0 = p[0] - self._start[0]
            x1 = p[1] - self._start[1]
            x2 = p[2] - self._start[2]

            return int(x2 + self.local_size[2] * (
                x1 + self.local_size[1] * x0))
        else:
            return False

    def __getitem__(self, item):
        return self.data[item]
    def __setitem__(self, key, value):
        self.data[key] = value


class OctalDataTree(object):
    def __init__(self, tree, ncomp, mode=None, dtype=ctypes.c_double):
        """
        Attach data to an OctalTree.
        :param tree: octal tree to use.
        :param ncomp: number of components per cell.
        :param mode: 'plain', 'halo' or 'parent'. 'plain' assigns ncomp to 
        each cube. 'halo' like 'plain' but with space for halo data. 'parent' 
        allocates to level l the cell count of level l-1.
        :param dtype: data type of elements.
        """
        if not mode in ('plain', 'halo', 'parent'):
            raise RuntimeError('bad mode passed')
        if ncomp != int(ncomp) or ncomp < 1:
            raise RuntimeError('bad ncomp passed')

        self.tree = tree
        self.ncomp = ncomp
        self.dtype = dtype
        self.mode = mode
        self.data = []
        self.num_data = []

        for lvl in self.tree.levels:
            if self.mode == 'plain' and \
                lvl.local_grid_cube_size is not None:
                    shape = list(lvl.local_grid_cube_size) + [ncomp]
            elif self.mode == 'halo' and \
                lvl.grid_cube_size is not None:
                    shape = list(lvl.grid_cube_size) + [ncomp]
            elif self.mode == 'parent' and \
                lvl.parent_local_size is not None:
                    shape = list(lvl.parent_local_size) + [ncomp]
            else:
                shape = (0,0,0,0)
            self.data.append(np.zeros(shape=shape, dtype=dtype))
            self.num_data.append(shape[0]*shape[1]*shape[2]*shape[3])

        self.nbytes = sum([dx.nbytes for dx in self.data])

    def halo_exchange_level(self, level):
        if level < 1:
            raise RuntimeError('Cannot exchange levels < 1')
        elif level >= self.tree.num_levels:
            raise RuntimeError('Cannot exchange levels > {}'.format(
                self.tree.num_levels-1))
        elif self.mode != 'halo':
            raise RuntimeError("Can only halo exchange if mode == halo")

        self.tree.levels[level].halo_exchange(self.data[level])

    def __getitem__(self, item):
        return self.data[item]


def send_parent_to_halo(src_level, parent_data_tree, halo_data_tree):
    """
    Copy the data from parent mode OctalDataTree to halo mode OctalDataTree.
    
    This function enables the movement of data up the octal tree. Data is sent
    from src_level to src_level - 1.
    
    :param src_level: parent level to send into the halo level
    :param parent_data_tree: OctalDataTree of mode parent.
    :param halo_data_tree: OctalDataTree of mode halo
    :return: halo_data_tree will be modified.
    """

    if halo_data_tree.ncomp != parent_data_tree.ncomp:
        raise RuntimeError('number of components is not consistent between' +\
                           ' trees')
    if src_level < 1 or src_level >= parent_data_tree.tree.num_levels:
        raise RuntimeError('bad src_level passed: {}'.format(src_level))
    if halo_data_tree.tree is not parent_data_tree.tree:
        raise RuntimeError('Passed OctalDataTree instances are not defined' +\
                           ' on the same OctalTree')
    if halo_data_tree.tree[src_level].comm is MPI.COMM_NULL:
        return

    # TODO: move this to C once working.
    tree = halo_data_tree.tree
    ns = tree[src_level - 1].ncubes_side_global
    ncomp = parent_data_tree.ncomp
    # use the cart_comm on the fine level.
    comm = tree[src_level].comm
    rank = comm.Get_rank()

    dst = halo_data_tree[src_level - 1].ravel()
    src = parent_data_tree[src_level].ravel()

    dst_g2l = tree[src_level - 1].global_to_local_halo
    src_g2l = tree[src_level].global_to_local_parent

    src_owners = tree[src_level].owners.ravel()
    dst_owners = tree[src_level-1].owners.ravel()

    send_req = None
    recv_req = None

    for cxt in itertools.product(range(ns), range(ns), range(ns)):

        cx = cube_tuple_to_lin_zyx(cxt, ns)

        cxc = cube_tuple_to_lin_zyx((cxt[0]*2, cxt[1]*2, cxt[2]*2), ns*2)
        dst_owner = dst_owners[cx]
        src_owner = src_owners[cxc]

        # TODO: trim excess looping here with more refined maps
        if src_owner != rank and dst_owner != rank:
            continue
        if src_owner == rank and dst_owner == rank:
            # can do direct copy
            src_index_b = src_g2l[cxt]*ncomp
            src_index_e = src_index_b + ncomp

            dst_index_b = dst_g2l[cxt]*ncomp
            dst_index_e = dst_index_b + ncomp

            dst[dst_index_b:dst_index_e:] = src[src_index_b: src_index_e:]

        elif src_owner == rank:
            # we are sending
            index_b = src_g2l[cxt[0], cxt[1], cxt[2]]*ncomp
            index_e = ncomp + index_b
            if send_req is not None: send_req.wait()
            send_req = comm.Isend(src[index_b:index_e:], dst_owner, tag=cx)
        elif dst_owner == rank:
            # we are recving
            index_b = dst_g2l[cxt]*ncomp
            index_e = ncomp + index_b
            if recv_req is not None: recv_req.wait()
            recv_req = comm.Irecv(dst[index_b:index_e:], src_owner, tag=cx)
        else:
            raise RuntimeError('Unknown data movement error.')

    if send_req is not None: send_req.wait()
    if recv_req is not None: recv_req.wait()


def send_plain_to_parent(src_level, plain_data_tree, parent_data_tree):
    """
    Copy the data from plain mode OctalDataTree to parent mode OctalDataTree.
    
    This function enables the movement of data down the octal tree. Data is 
    sent from src_level to src_level + 1.
    
    :param src_level: plain level to send into the parent level
    :param plain_data_tree: OctalDataTree of mode plain.
    :param parent_data_tree: OctalDataTree of mode parent
    :return: parent_data_tree will be modified.
    """

    if parent_data_tree.ncomp != plain_data_tree.ncomp:
        raise RuntimeError('number of components is not consistent between' +\
                           ' trees')
    if src_level < 0 or src_level >= plain_data_tree.tree.num_levels-1:
        raise RuntimeError('bad src_level passed: {}'.format(src_level))
    if parent_data_tree.tree is not plain_data_tree.tree:
        raise RuntimeError('Passed OctalDataTree instances are not defined' +\
                           ' on the same OctalTree')
    if parent_data_tree.tree[src_level+1].comm is MPI.COMM_NULL:
        return

    # TODO: move this to C once working.
    tree = parent_data_tree.tree
    ns = tree[src_level].ncubes_side_global
    ncomp = plain_data_tree.ncomp
    # use the cart_comm on the fine level.
    comm = tree[src_level+1].comm
    rank = comm.Get_rank()

    dst = parent_data_tree[src_level + 1].ravel()
    src = plain_data_tree[src_level].ravel()

    dst_g2l = tree[src_level + 1].global_to_local_parent
    src_g2l = tree[src_level].global_to_local

    src_owners = tree[src_level].owners.ravel()
    dst_owners = tree[src_level+1].owners.ravel()

    send_req = None
    recv_req = None

    for cxt in itertools.product(range(ns), range(ns), range(ns)):

        cx = cube_tuple_to_lin_zyx(cxt, ns)
        cxc = cube_tuple_to_lin_zyx((cxt[0]*2, cxt[1]*2, cxt[2]*2), ns*2)

        dst_owner = dst_owners[cxc]
        src_owner = src_owners[cx]

        # TODO: trim excess looping here with more refined maps
        if src_owner != rank and dst_owner != rank:
            continue
        if src_owner == rank and dst_owner == rank:
            # can do direct copy
            src_index_b = src_g2l[cxt]*ncomp
            src_index_e = src_index_b + ncomp

            dst_index_b = dst_g2l[cxt]*ncomp
            dst_index_e = dst_index_b + ncomp

            dst[dst_index_b:dst_index_e:] = src[src_index_b: src_index_e:]

        elif src_owner == rank:
            # we are sending
            index_b = src_g2l[cxt[0], cxt[1], cxt[2]]*ncomp
            index_e = ncomp + index_b

            if send_req is not None: send_req.wait()
            send_req = comm.Isend(src[index_b:index_e:], dst_owner, tag=cx)
        elif dst_owner == rank:
            # we are recving
            index_b = dst_g2l[cxt]*ncomp

            index_e = ncomp + index_b
            if recv_req is not None: recv_req.wait()
            recv_req = comm.Irecv(dst[index_b:index_e:], src_owner, tag=cx)
        else:
            raise RuntimeError('Unknown data movement error.')

    if send_req is not None: send_req.wait()
    if recv_req is not None: recv_req.wait()





