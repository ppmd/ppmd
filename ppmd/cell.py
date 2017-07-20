from __future__ import print_function, division, absolute_import


__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes as ct
import math
import numpy as np
import itertools

# package level imports
from ppmd import host, runtime, kernel, opt
import ppmd.lib.shared_lib

def radius_cell_decompose(rc, rd, verbose=False):
    """
    returns list of cell offset tuples for cell sub-division with cell width
    rd matching a interaction cutoff rc.
    """

    rc = float(rc)
    rd[0] = float(rd[0])
    rd[1] = float(rd[1])
    rd[2] = float(rd[2])

    rc2 = rc*rc

    maxdx = int(math.ceil(rc/rd[0])) + 1
    maxdy = int(math.ceil(rc/rd[1])) + 1
    maxdz = int(math.ceil(rc/rd[2])) + 1

    argx = (-1*maxdx, maxdx+1)
    argy = (-1*maxdy, maxdy+1)
    argz = (-1*maxdz, maxdz+1)


    offsets2 = []

    # create the axis directions first as moving along the axis is tedious
    offsets2 += [(ix,0, 0) for ix in range(*argx)] + \
        [(0, ix, 0) for ix in range(*argy)] + \
        [(0, 0, ix) for ix in range(*argz)]
        
    # planes
    for ix in itertools.product([0], range(1, maxdy), range(1, maxdz)):
        if ((ix[1]-1)*rd[1])**2. + ((ix[2]-1)*rd[2])**2. < rc2:
            offsets2 += [(ix[0], ix[1]*s[0], ix[2]*s[1]) for s in itertools.product([-1, 1], [-1, 1]) ]

    for ix in itertools.product( range(1, maxdx), [0], range(1, maxdz)):
        if ((ix[0]-1)*rd[0])**2. + ((ix[2]-1)*rd[2])**2. < rc2:
            offsets2 += [(ix[0]*s[0], ix[1], ix[2]*s[1]) for s in itertools.product([-1, 1], [-1, 1]) ]

    for ix in itertools.product( range(1, maxdx), range(1, maxdy), [0]):
        if ((ix[0]-1)*rd[0])**2. + ((ix[1]-1)*rd[1])**2. < rc2:
            offsets2 += [(ix[0]*s[0], ix[1]*s[1], ix[2]) for s in itertools.product([-1, 1], [-1, 1]) ]

    # quadrants
    for ix in itertools.product(range(1, maxdx), range(1, maxdy), range(1, maxdz)):
        if ((ix[0]-1)*rd[0])**2. + ((ix[1]-1)*rd[0])**2. + ((ix[2]-1)*rd[0])**2. < rc2:
            offsets2 += [
                (ix[0]*s[0], ix[1]*s[1], ix[2]*s[2]) for s in itertools.product(
                    [-1, 1], [-1, 1], [-1, 1]
                )
            ]

    return set(offsets2)

def convert_offset_tuples(offsets, cell_array, remove_zero=False):
    dy = cell_array[0]
    dz = dy*cell_array[1]

    a = set([ ix[0] + ix[1]*dy + ix[2]*dz for ix in offsets])
    if remove_zero:
        a.remove(0)
    return sorted(a)


class CellList(object):
    """
    Class to handle cell lists for a given domain.
    """

    def __init__(self, n_func, positions, domain):
        """
        Setup the cell list with a set of positions and a domain.
        :param n_func: Function handle to get number of local particles.
        :param positions: Positions to use to sort into cells using.
        :param domain: Domain to setup with cell array.
        :param cell_width: Cell width to use for domain partitioning.
        :return:
        """
        self._n = n_func
        self._positions = positions
        self._domain = domain

        # Is class initialised?
        self._init = False

        # Timer inits
        self.timer_sort = opt.Timer()

        # container for cell list.
        self._cell_list = None

        # contents count for each cell.
        self._cell_contents_count = None

        # container for reverse lookup. (If needed ?)
        self._cell_reverse_lookup = None


        # static args init.
        self._static_args = None
        self._cell_sort_lib = None
        self._halo_cell_sort_loop = None

        self.halos_exist = True

        self.update_required = True

        self.version_id = 0
        """Version id, incremented when the list is updated."""

        self.halo_version_id = 0
        """halo version id incremented when halo cell list is updated."""

        # vars for automatic updating based on a counter

        self._update_set = False
        self._update_func = None
        self._update_func_pre = None
        self._update_func_post = None

        self.create()

    def reset_callbacks(self):
        self._update_func = None
        self._update_func_pre = None
        self._update_func_post = None

    def get_setup_parameters(self):
        """
        Get the values/function handles used to setup the cell list.
        :return:
        """

        assert (None in (self._n, self._positions, self._domain)) is False, "get_setup_parameters Error: cell list not setup."
        return self._n, self._positions, self._domain


    def create(self):
        self._cell_sort_setup()


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

    def _update_tracking(self):

        if self._update_func is None:
            return True

        if self._update_set and self._update_func():
            return True
        else:
            return False

    def setup_pre_update(self, func):
        self._update_func_pre = func


    def _pre_update(self):
        """
        Run a pre update function eg boundary conditions.
        """
        if self._update_func_pre is not None:
            self._update_func_pre()
            # pass


    def check(self):
        """
        Check if the cell_list needs updating and update if required.
        :return:
        """
        if not self._init:
            self._cell_sort_setup()

            if not self._init:
                print("Initalisation failed")
                return False


        if (self.update_required is True) or self._update_tracking():
            self._pre_update()

            self.sort()
            if self._update_func_post is not None:
                self._update_func_post()
            return True
        else:
            return False


    def _cell_sort_setup(self):
        """
        Creates looping for cell list creation
        """

        '''Construct initial cell list'''
        self._cell_list = host.Array(dtype=ct.c_int, ncomp=self._positions.max_npart + self._domain.cell_count + 1)

        '''Keep track of number of particles per cell'''
        self._cell_contents_count = host.Array(np.zeros([self._domain.cell_count]), dtype=ct.c_int)

        '''Reverse lookup, given a local particle id, get containing cell.'''
        self._cell_reverse_lookup = host.Array(dtype=ct.c_int, ncomp=self._positions.max_npart)
        
        

        _cell_sort_code = '''

        int ix;

        const double _icel0 = 1.0/CEL[0];
        const double _icel1 = 1.0/CEL[1];
        const double _icel2 = 1.0/CEL[2];

        const double _b0 = B[0];
        const double _b2 = B[2];
        const double _b4 = B[4];


        for (ix=0; ix<end_ix; ix++) {

        int C0 = 1 + (int)((P[ix*3]     - _b0)*_icel0);
        int C1 = 1 + (int)((P[ix*3 + 1] - _b2)*_icel1);
        int C2 = 1 + (int)((P[ix*3 + 2] - _b4)*_icel2);


        if ((C0 < 1) || (C0 > (CA[0]-2))) {

            if ( (C0 > (CA[0]-2)) && (P[ix*3] <= B[1]  )) {

                C0 = CA[0]-2;

            } else {
                cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C0 " << C0 << endl;
                cout << "B[0] " << B[0] << " B[1] " << B[1] << " Px " << P[ix*3+0] << " " << (P[ix*3]-_b0)*_icel0 << endl;
            }
        }
        if ((C1 < 1) || (C1 > (CA[1]-2))) {

            if ( (C1 > (CA[1]-2)) && (P[ix*3+1] <= B[3]  )) {

                C1 = CA[1]-2;

            } else {

                cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C1 " << C1 << endl;
                cout << "B[2] " << B[2] << " B[3] " << B[3] << " Py " << P[ix*3+1]<< " " << (P[ix*3+1]-_b2)*_icel1 << endl;
            }
        }
        if ((C2 < 1) || (C2 > (CA[2]-2))) {

            if ( (C2 > (CA[2]-2)) && (P[ix*3+2] <= B[5]  )) {

                C2 = CA[2]-2;

            } else {

                cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C2 " << C2 << endl;
                cout << "B[4] " << B[4] << " B[5] " << B[5] << " Pz " << P[ix*3+2]<< " " << (P[ix*3 + 2]-_b4)*_icel2 << endl;
            }
        }


        const int val = (C2*CA[1] + C1)*CA[0] + C0;
        CCC[val]++;
        CRL[ix] = val;
        

        q[ix] = q[n + val];
        q[n + val] = ix;

        }
        '''
        _dat_dict = {'B': self._domain.boundary,              # Inner boundary on local domain (inc halo cells)
                     'P': self._positions,                    # positions
                     'CEL': self._domain.cell_edge_lengths,   # cell edge lengths
                     'CA': self._domain.cell_array,           # local domain cell array
                     'q': self._cell_list,                    # cell list
                     'CCC': self._cell_contents_count,        # contents count for each cell
                     'CRL': self._cell_reverse_lookup}        # reverse cell lookup map

        _static_args = {'end_ix': ct.c_int,  # Number of particles.
                        'n': ct.c_int}       # start of cell point in list.



        _cell_sort_kernel = kernel.Kernel('cell_class_cell_list_method', _cell_sort_code, headers=['stdio.h'], static_args=_static_args)
        self._cell_sort_lib = ppmd.lib.shared_lib.SharedLib(_cell_sort_kernel, _dat_dict)

        self._init = True

    def trigger_update(self):
        """
        Trigger an update of the cell list.
        """
        self.update_required = True


    def sort(self):
        """
        Sort local particles into cell list.
        :return:
        """
        if self._init:

            _dat_dict = {'B': self._domain.boundary,
                         'P': self._positions,
                         'CEL': self._domain.cell_edge_lengths,
                         'CA': self._domain.cell_array,
                         'q': self._cell_list,
                         'CCC': self._cell_contents_count,
                         'CRL': self._cell_reverse_lookup}


            self.timer_sort.start()

            _n = self._cell_list.end - self._domain.cell_count

            self._cell_list[self._cell_list.end] = _n
            self._cell_list.data[_n:self._cell_list.end:] = ct.c_int(-1)
            self._cell_contents_count.zero()

            self._cell_sort_lib.execute(
                static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)},
                dat_dict=_dat_dict
            )

            self.version_id += 1
            self.update_required = False

            self.timer_sort.pause()

            opt.PROFILE[
                self.__class__.__name__+':sort'
            ] = (self.timer_sort.time())


        else:
            print("CELL LIST NOT INITIALISED")


    @property
    def cell_list(self):
        """
        :return: The held cell list.
        """
        return self._cell_list

    @property
    def offset(self):
        """
        Get the offset required to find the starting position of the cells in the cell list.

        :return: int start of cells in cell list.
        """
        return ct.c_int(self._cell_list.end - self._domain.cell_count)

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

    @property
    def max_cell_contents_count(self):
        return np.max(self.cell_contents_count[:])

    @property
    def domain(self):
        """
        :return: The domain used.
        """
        return self._domain

    def _setup_halo_sorting_lib(self):
        """
        Setup the library to sort particles after halo transfer.
        """
        _cell_sort_code = '''

        int index = shift;
        for(int ix = 0; ix < CC; ix++){

            //get number of particles
            const int _tmp = CRC[ix];

            if (_tmp>0){

                //first index in cell region of cell list.
                q[end+LCI[ix]] = index;
                CCC[LCI[ix]] = _tmp;

                //start at first particle in halo cell, work forwards
                for(int iy = 0; iy < _tmp-1; iy++){
                    q[index+iy]=index+iy+1;
                    CRL[index+iy]=LCI[ix];
                }
                q[index+_tmp-1] = -1;
                CRL[index+_tmp-1] = LCI[ix];
            }


            index += CRC[ix];
        }
        '''

        _static_args = {'CC': ct.c_int, 'shift': ct.c_int, 'end': ct.c_int}

        _cell_sort_dict = {
            'q': host.NullIntArray,
            'LCI': host.NullIntArray,
            'CRC': host.NullIntArray,
            'CCC': host.NullIntArray,
            'CRL': self.cell_reverse_lookup
        }

        _cell_sort_kernel = kernel.Kernel('halo_cell_list_method', _cell_sort_code, headers=['stdio.h'],
                                          static_args=_static_args)
        self._halo_cell_sort_loop = ppmd.lib.shared_lib.SharedLib(_cell_sort_kernel, _cell_sort_dict)


    def prepare_halo_sort(self, total_size):

        # if the total size is larger than the current array we need to resize.

        cell_count = self._domain.cell_array[0] * self._domain.cell_array[1] * self._domain.cell_array[2]

        if total_size + cell_count + 1 > self._cell_list.ncomp:
            cell_start = self._cell_list[self._cell_list.end]
            cell_end = self._cell_list.end

            self._cell_list.realloc(total_size + cell_count + 1)

            self._cell_list.data[self._cell_list.end - cell_count: self._cell_list.end:] = self._cell_list.data[cell_start:cell_end:]

            self._cell_list.data[self._cell_list.end] = self._cell_list.end - cell_count

            # cell reverse lookup
            self._cell_reverse_lookup.realloc(total_size)


    def post_halo_exchange(self):

        self.halo_version_id += 1



    def sort_halo_cells(self,local_cell_indices_array, cell_contents_recv, npart, total_size):

        # if the total size is larger than the current array we need to resize.


        self.timer_sort.start()

        cell_count = self._domain.cell_array[0] * self._domain.cell_array[1] * self._domain.cell_array[2]

        if total_size + cell_count + 1 > self._cell_list.ncomp:
            cell_start = self._cell_list[self._cell_list.end]
            cell_end = self._cell_list.end

            self._cell_list.realloc(total_size + cell_count + 1)

            self._cell_list.data[self._cell_list.end - cell_count: self._cell_list.end:] = self._cell_list.data[cell_start:cell_end:]

            self._cell_list.data[self._cell_list.end] = self._cell_list.end - cell_count

            # cell reverse lookup
            self._cell_reverse_lookup.realloc(total_size)

        if self._halo_cell_sort_loop is None:
            self._setup_halo_sorting_lib()

        _cell_sort_dict = {
            'q': self._cell_list,
            'LCI': local_cell_indices_array,
            'CRC': cell_contents_recv,
            'CCC': self._cell_contents_count,
            'CRL': self.cell_reverse_lookup
        }

        _cell_sort_static_args = {'CC': ct.c_int(cell_contents_recv.ncomp),
                                  'shift': ct.c_int(npart),
                                  'end': ct.c_int(self._cell_list[self._cell_list.end])}
        
        self._halo_cell_sort_loop.execute(static_args=_cell_sort_static_args,
                                          dat_dict=_cell_sort_dict)
        
        self.halo_version_id += 1

        self.timer_sort.pause()


    @property
    def num_particles(self):
        """
        Get the number of particles in the cell list
        """
        return self._n()

    @property
    def total_num_particles(self):
        """
        Number of local particles + halo particles.
        """
        return self._positions.npart_total

    @property
    def cell_width(self):
        """
        Return the cell width used to setup the cell structure. N.B. cells may be larger than this.
        """
        return self._domain.cell_edge_lengths[0]

