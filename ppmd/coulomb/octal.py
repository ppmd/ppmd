from __future__ import print_function, division, absolute_import

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
    """
    def __init__(self, cart_comm, cube_side_count):

        self.cart_comm = cart_comm
        topo = self.cart_comm.Get_topo()
        dims= topo[0][::-1]
        self.cube_count = cube_side_count**len(dims)
        self.cube_side_count = cube_side_count

        # self.compute_grid_ownership(dims, cube_side_count)

    @staticmethod
    def compute_grid_ownership(dims, cube_side_count):

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









