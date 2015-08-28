# cell list container
import data
import ctypes as ct
import numpy as np
import gpucuda
import build
import kernel


class CellList(object):
    """
    Class to handle cell lists for a given domain.
    """

    def __init__(self):
        # Is class initialised?
        self._init = False

        # container for cell list.
        self._cell_list = None

        # contents count for each cell.
        self._cell_contents_count = None

        # container for reverse lookup. (If needed)
        self._cell_reverse_lookup = None

        # domain to partition.
        self._domain = None

        # positions init
        self._positions = None

        #function handle to get number of local particles from state
        self._n = None

        # static args init.
        self._static_args = None
        self._cell_sort_lib = None

    def setup(self, n, positions, domain, cell_width):
        """
        Setup the cell list with a set of positions and a domain.
        :param n: Function handle to get number of local particles.
        :param positions: Positions to use to sort into cells using.
        :param domain: Domain to setup with cell array.
        :param cell_width: Cell width to use for domain partitioning.
        :return:
        """
        self._n = n
        self._positions = positions
        self._domain = domain

        # partition domain.
        self._domain.set_cell_array_radius(cell_width)

        # setup methods to sort into cells.
        self._cell_sort_setup()

    def _cell_sort_setup(self):
        """
        Creates looping for cell list creation
        """

        '''Construct initial cell list'''
        self._cell_list = data.ScalarArray(dtype=ct.c_int, max_size=self._positions.max_size + self._domain.cell_count + 1)

        '''Keep track of number of particles per cell'''
        self._cell_contents_count = data.ScalarArray(np.zeros([self._domain.cell_count]), dtype=ct.c_int)

        '''Reverse lookup, given a local particle id, get containing cell.'''
        self._cell_reverse_lookup = data.ScalarArray(dtype=ct.c_int, max_size=self._positions.max_size)

        # add gpu arrays
        if gpucuda.INIT_STATUS():
            self._cell_reverse_lookup.add_cuda_dat()
            self._cell_contents_count.add_cuda_dat()
            self._cell_list.add_cuda_dat()


        _cell_sort_code = '''

        int ix;
        for (ix=0; ix<end_ix; ix++) {

        const int C0 = (int)((P[ix*3]     - B[0])/CEL[0]);
        const int C1 = (int)((P[ix*3 + 1] - B[2])/CEL[1]);
        const int C2 = (int)((P[ix*3 + 2] - B[4])/CEL[2]);

        const int val = (C2*CA[1] + C1)*CA[0] + C0;

        //needed, may improve halo exchange times
        CCC[val]++;
        CRL[ix] = val;

        q[ix] = q[n + val];
        q[n + val] = ix;

        }
        '''
        _dat_dict = {'B': self._domain.boundary_outer,        # Outer boundary on local domain (inc halo cells)
                     'P': self._positions,                    # positions
                     'CEL': self._domain.cell_edge_lengths,   # cell edge lengths
                     'CA': self._domain.cell_array,           # local domain cell array
                     'q': self._cell_list,                    # cell list
                     'CCC': self._cell_contents_count,        # contents count for each cell
                     'CRL': self._cell_reverse_lookup}        # reverse cell lookup map

        _static_args = {'end_ix': ct.c_int,  # Number of particles.
                        'n': ct.c_int}       # start of cell point in list.


        _cell_sort_kernel = kernel.Kernel('cell_class_cell_list_method', _cell_sort_code, headers=['stdio.h'], static_args=_static_args)
        self._cell_sort_lib = build.SharedLib(_cell_sort_kernel, _dat_dict)

    def sort(self):
        """
        Sort local particles into cell list.
        :return:
        """
        _n = self._cell_list.end - self._domain.cell_count

        self._cell_list[self._cell_list.end] = _n
        self._cell_list.dat[_n:self._cell_list.end:] = ct.c_int(-1)
        self._cell_contents_count.zero()

        self._cell_sort_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

    @property
    def cell_list(self):
        """
        :return: The held cell list.
        """
        return self._cell_list

    @property
    def cell_reverse_lookup(self):
        """
        :return: The Reverse lookup map.
        """
        return self._cell_reverse_lookup

    @property
    def cell_contents_count(self):
        """
        :return: The cell contents count lookup
        """
        return self._cell_contents_count




# default cell list
cell_list = CellList()
