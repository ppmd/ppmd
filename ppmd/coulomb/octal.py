from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import itertools, ctypes
import numpy as np

class cube_owner_map(object):
    """
    Compute the global map from cube index (lexicographic) to owning MPI rank
    and map to MPI ranks with sub-domains of non-empty intersection with each
    cube.
    
    Assumes that in each dimension ranks are assigned equal shares of the 
    domain.
    
    :param cart_comm: MPI cartesian communicator.
    :param cube_side_count: Integer subdivision level of domain.
    """
    def __init__(self, cart_comm, cube_side_count):

        self.cart_comm = cart_comm
        topo = self.cart_comm.Get_topo()
        dims= topo[0]
        self.cube_count = cube_side_count**len(dims)
        self.cube_side_count = cube_side_count
        owners, contribs = self.compute_grid_ownership(dims, cube_side_count)

    @staticmethod
    def compute_grid_ownership(dims, cube_side_count):
        """
        For each dimension compute the owner and contributing ranks. Does
        not perform out product to compute the full map.
        :param dims: tuple of mpi dims.
        :param cube_side_count: int, number of cubes in each dimension
        :return: tuple: ranks of each cube owner, contributing ranks.
        """

        ndim = len(dims)
        oocsc = 1.0 / cube_side_count
        hoocsc = 0.5 / cube_side_count
        cube_mids = [ hoocsc + oocsc*ix for ix in range(cube_side_count) ]
        cube_lower_edge = lambda x: x - hoocsc
        cube_upper_edge = lambda x: x + hoocsc

        dim_owners = [[-1 for iy in range(cube_side_count)] for
                      ix in range(ndim)]

        dim_contribs = [[[] for iy in range(cube_side_count) ] for
                        ix in range(ndim)]

        for dx in range(ndim):
            lbound = lambda argx: float(argx)/dims[dx]
            ubound = lambda argx: float(argx+1)/dims[dx]
            for mx in range(dims[dx]):
                for cx, cmid in enumerate(cube_mids):
                    if (lbound(mx) <= cmid) and (cmid < ubound(mx)):
                        # this rank is the cube's unique owner
                        dim_owners[dx][cx] = mx

                    elif (ubound(mx) > cube_lower_edge(cmid)) and \
                            (lbound(mx) < cube_upper_edge(cmid)):
                        # this rank's subdomain intersects the cube's lower
                        # bound
                        dim_contribs[dx][cx].append(mx)

                    elif (lbound(mx) < cube_upper_edge(cmid)) and \
                            (ubound(mx) > cube_lower_edge(cmid)):
                        # rank intersects cube's upper edge
                        dim_contribs[dx][cx].append(mx)

        return dim_owners, dim_contribs


    @staticmethod
    def compute_map_product(cart_comm, dim_owners, dim_contribs):
        """
        Compute the full map from cube index to mpi rank owner, cube index to
        mpi ranks of contributors.
        :param: MPI cart_comm to use for logical coord to rank conversion.
        :param dim_owners: ndims x cube_side_count tuple representing cube to 
        owner map per dimension, elements are ints.
        :param dim_contribs: tuple ndims x cube_side_count tuple containing
        lists of contributors.
        :return: full map.
        """
        # domain is a cube therefore we can inspect one dimension
        ncubes_per_side = len(dim_owners[0])
        ndim = len(dim_owners)
        ncubes = ncubes_per_side**ndim
        iterset = [range(ncubes_per_side) for i in range(ndim)]
        cube_to_mpi = np.zeros(ncubes, dtype=ctypes.c_uint64)

        tuple_to_lin_coeff = [
            ncubes_per_side**(ndim-dx-1) for dx in range(ndim)]

        cube_tuple_to_lin = lambda X: sum([i[0]*i[1] for i in zip(
            X,tuple_to_lin_coeff)])

        # loop over cube indexes
        for cube_tuple in itertools.product(*iterset):
            cx = cube_tuple_to_lin(cube_tuple)
            ompi = cart_comm.Get_cart_rank([
                dim_owners[i][cube_tuple[i]] for i in range(ndim)])
            cube_to_mpi[cx] = ompi

        return cube_to_mpi



    @property
    def get_cube_to_owner(self):
        """
        :return: Array ctypes.c_int, index i contains MPI rank owning cube i.
        """
        return

    @property
    def get_cube_to_contributors(self):
        """
        :return: tuple: [0] Array length ncubes+1 of start points in [1] 
        array of contributors.
        """
        return









