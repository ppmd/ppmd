# cell list container
import host
import mpi
import runtime
import ctypes as ct
import numpy as np
import build
import kernel
import math


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

        # container for reverse lookup. (If needed ?)
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
        self._halo_cell_sort_loop = None

        self.halos_exist = False

        self.version_id = 0
        """version id, incremented when the list is updated."""


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
        _err = self._domain.set_cell_array_radius(cell_width)

        # setup methods to sort into cells.
        self._cell_sort_setup()

        if (_err is True) and (self._domain.halos is not False):
            self.halos_exist = True

        return _err

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

        self.version_id += 1


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
                }
                q[index+_tmp-1] = -1;
            }


            index += CRC[ix];
        }
        '''

        _static_args = {'CC': ct.c_int, 'shift': ct.c_int, 'end': ct.c_int}

        _cell_sort_dict = {
            'q': host.NullIntArray,
            'LCI': host.NullIntArray,
            'CRC': host.NullIntArray,
            'CCC': host.NullIntArray
        }

        _cell_sort_kernel = kernel.Kernel('halo_cell_list_method', _cell_sort_code, headers=['stdio.h'],
                                          static_args=_static_args)
        self._halo_cell_sort_loop = build.SharedLib(_cell_sort_kernel, _cell_sort_dict)

    def sort_halo_cells(self,local_cell_indices_array, cell_contents_recv, npart):

        if self._halo_cell_sort_loop is None:
            self._setup_halo_sorting_lib()

        _cell_sort_dict = {
            'q': self._cell_list,
            'LCI': local_cell_indices_array,
            'CRC': cell_contents_recv,
            'CCC': self._cell_contents_count
        }

        _cell_sort_static_args = {'CC': ct.c_int(cell_contents_recv.ncomp),
                                  'shift': ct.c_int(npart),
                                  'end': ct.c_int(self._cell_list[self._cell_list.end])}

        self._halo_cell_sort_loop.execute(static_args=_cell_sort_static_args,
                                          dat_dict=_cell_sort_dict)


# default cell list
cell_list = CellList()


################################################################################################################
# GroupByCell definition
################################################################################################################


class GroupByCell(object):
    """
    Class to group dats based on the cells particles reside in.
    """
    def __init__(self):
        self._state = None
        self._new_particle_dats = None
        self._sizes = None
        self._cell_list_new = None
        self._group_by_cell_lib = None
        self.swaptimer = None

    def setup(self, state_in):
        """
        Setup library to group data in the state particle dats such that data for particles in the same cell are sequential.
        :param state state_in: State containing particle dats.
        """



        self._state = state_in
        self._new_particle_dats = []
        self._sizes = []

        for ix in self._state.particle_dats:
            _dat = getattr(self._state, ix)
            self._new_particle_dats.append(host.Matrix(nrow=_dat.max_npart, ncol=_dat.ncomp, dtype=_dat.dtype))
            self._sizes.append(_dat.ncomp)

        self._cell_list_new = host.Array(ncomp=cell_list.cell_list.ncomp, dtype=ct.c_int)

        if cell_list.domain.halos is not False:
            _triple_loop = 'for(int iz = 1; iz < (CA[2]-1); iz++){' \
                           'for(int iy = 1; iy < (CA[1]-1); iy++){ ' \
                           'for(int ix = 1; ix < (CA[0]-1); ix++){'
        else:
            _triple_loop = 'for(int iz = 0; iz < CA[2]; iz++){' \
                           'for(int iy = 0; iy < CA[1]; iy++){' \
                           'for(int ix = 0; ix < CA[0]; ix++){'


        # Create code for all particle dats.
        _dynamic_dats = ''
        _space = ' ' * 16
        for ix, iy in zip(self._state.particle_dats, self._sizes):
            if iy > 1:
                _dynamic_dats += _space + 'for(int ni = 0; ni < %(NCOMP)s; ni++){ \n' % {'NCOMP':iy}
                _dynamic_dats += _space + '%(NAME)s_new[(index*%(NCOMP)s)+ni] = %(NAME)s[(i*%(NCOMP)s)+ni]; \n' % {'NCOMP':iy, 'NAME':str(ix)}
                _dynamic_dats += _space + '} \n'
            else:
                _dynamic_dats += _space + '%(NAME)s_new[index] = %(NAME)s[i]; \n' % {'NAME':str(ix)}

        _code = '''
        int index = 0;

        %(TRIPLE_LOOP)s

            const int c = iz*CA[0]*CA[1] + iy*CA[0] + ix;

            int i = q[n + c];
            if (i > -1) { q_new[n + c] = index; }

            while (i > -1){
                //dynamic dats

                \n%(DYNAMIC_DATS)s

                // fixed parts, cell lists etc.
                PCL[index] = c;

                i = q[i];
                if (i > -1) { q_new[index] = index+1; } else { q_new[index] = -1; }

                index++;
            }
        }}}
        ''' % {'TRIPLE_LOOP': _triple_loop, 'DYNAMIC_DATS':_dynamic_dats}


        _static_args = {
            'n': ct.c_int
        }

        # Always needed arguments
        _args = {
            'CA': cell_list.domain.cell_array,
            'q': cell_list.cell_list,
            'q_new': self._cell_list_new,
            'PCL': cell_list.cell_reverse_lookup
        }

        # Dynamic arguments dependant on how many particle dats there are.
        for idx, ix in enumerate(self._state.particle_dats):
            # existing dat in state
            _args['%(NAME)s' % {'NAME':ix}] = getattr(self._state, ix)
            # new dat to copy into
            _args['%(NAME)s_new' % {'NAME':ix}] = self._new_particle_dats[idx]


        _headers = ['stdio.h']

        _name = ''
        for ix in self._state.particle_dats:
            _name += '_' + str(ix)

        _kernel = kernel.Kernel('CellGroupCollect' + _name, _code, None, _headers, None, _static_args)
        self._group_by_cell_lib = build.SharedLib(_kernel, _args)
        self.swaptimer = runtime.Timer(runtime.TIMER, 0)


    def group_by_cell(self):
        """
        Run library to group data by cell.
        """

        self._cell_list_new.dat[cell_list.cell_list[cell_list.cell_list.end]:cell_list.cell_list.end:] = ct.c_int(-1)

        self._group_by_cell_lib.execute(static_args={'n':ct.c_int(cell_list.cell_list[cell_list.cell_list.end])})

        # swap array pointers

        self.swaptimer.start()

        # swap dynamic dats
        for idx, ix in enumerate(self._state.particle_dats):
            _tmp = getattr(self._state, ix).dat
            getattr(self._state, ix).dat = self._new_particle_dats[idx].dat
            self._new_particle_dats[idx].dat = _tmp


        # swap cell list.
        _tmp = cell_list.cell_list.dat
        cell_list.cell_list.dat = self._cell_list_new.dat
        self._cell_list_new.dat = _tmp
        cell_list.cell_list.dat[cell_list.cell_list.end] = cell_list.cell_list.end - cell_list.domain.cell_count

        self.swaptimer.pause()

group_by_cell = GroupByCell()


################################################################################################################
# NeighbourList definition
################################################################################################################


class NeighbourList(object):

    def __init__(self, list=cell_list):
        self.cell_list = cell_list
        self.max_len = None
        self.list = None
        self.lib = None

        self.version_id = 0
        """Update tracking of neighbour list. """

        self.cell_width = None
        self.max_traveled_distance = 0
        self.time = 0
        self._time_func = None


        self._positions = None
        self._velocities = None
        self._domain = None
        self.neighbour_starting_points = None
        self.cell_width_squared = None
        self._neighbour_lib = None
        self._n = None

        self.n_local = None
        self.n_total = None

        self._last_n = -1

        """Return the number of particle that have neighbours listed"""

        self._return_code = None

        self.update_required = True

    def setup(self, n, positions, velocities, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._velocities = velocities
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_int)

        _initial_factor = math.ceil(15. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))


        self.max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_int)

        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.dat[0] = -1

        _code = '''

        const double cutoff = CUTOFF[0];
        const int max_len = MAX_LEN[0];

        const int _h_map[27][3] = {
                                            {-1,1,-1},
                                            {-1,-1,-1},
                                            {-1,0,-1},
                                            {0,1,-1},
                                            {0,-1,-1},
                                            {0,0,-1},
                                            {1,0,-1},
                                            {1,1,-1},
                                            {1,-1,-1},

                                            {-1,1,0},
                                            {-1,0,0},
                                            {-1,-1,0},
                                            {0,-1,0},
                                            {0,0,0},
                                            {0,1,0},
                                            {1,0,0},
                                            {1,1,0},
                                            {1,-1,0},

                                            {-1,0,1},
                                            {-1,1,1},
                                            {-1,-1,1},
                                            {0,0,1},
                                            {0,1,1},
                                            {0,-1,1},
                                            {1,0,1},
                                            {1,1,1},
                                            {1,-1,1}
                                        };

        int tmp_offset[27];
        for(int ix=0; ix<27; ix++){
            tmp_offset[ix] = _h_map[ix][0] + _h_map[ix][1] * CA[0] + _h_map[ix][2] * CA[0]* CA[1];
        }

        // loop over particles
        int m = -1;
        for (int ix=0; ix<end_ix; ix++) {

            const int C0 = (int)((P[ix*3]     - B[0])/CEL[0]);
            const int C1 = (int)((P[ix*3 + 1] - B[2])/CEL[1]);
            const int C2 = (int)((P[ix*3 + 2] - B[4])/CEL[2]);

            const int val = (C2*CA[1] + C1)*CA[0] + C0;

            NEIGHBOUR_STARTS[ix] = m + 1;

            for(int k = 0; k < 27; k++){

                int iy = q[n + val + tmp_offset[k]];
                while (iy > -1) {
                    if (ix < iy){

                        const double rj0 = P[iy*3] - P[ix*3];
                        const double rj1 = P[iy*3+1] - P[ix*3 + 1];
                        const double rj2 = P[iy*3+2] - P[ix*3 + 2];

                        if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {
                            m++;
                            if (m < max_len){
                                NEIGHBOUR_LIST[m] = iy;
                            } else {
                                RC[0] = -1;
                                return;
                            }
                        }

                    }

                    iy = q[iy];
                }

            }
        }
        NEIGHBOUR_STARTS[end_ix] = m + 1;

        RC[0] = 0;
        return;
        '''
        _dat_dict = {'B': self._domain.boundary_outer,        # Outer boundary on local domain (inc halo cells)
                     'P': self._positions,                    # positions
                     'CEL': self._domain.cell_edge_lengths,   # cell edge lengths
                     'CA': self._domain.cell_array,           # local domain cell array
                     'q': self.cell_list.cell_list,           # cell list
                     'CUTOFF': self.cell_width_squared,
                     'NEIGHBOUR_STARTS': self.neighbour_starting_points,
                     'NEIGHBOUR_LIST': self.list,
                     'MAX_LEN': self.max_len,
                     'RC': self._return_code}

        _static_args = {'end_ix': ct.c_int,  # Number of particles.
                        'n': ct.c_int}       # start of cell point in list.


        _kernel = kernel.Kernel('cell_neighbour_list_method', _code, headers=['stdio.h'], static_args=_static_args)
        self._neighbour_lib = build.SharedLib(_kernel, _dat_dict)

    def trigger_update_required(self):
        self.update_required = True

    def update(self, _attempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."



        if self.update_required is True:
            if self.neighbour_starting_points.ncomp < self._n() + 1:
                self.neighbour_starting_points.realloc(self._n() + 1)
            if runtime.VERBOSE.level > 2:
                print "rank:", mpi.MPI_HANDLE.rank, "rebuilding neighbour list"


            _n = cell_list.cell_list.end - self._domain.cell_count
            self._neighbour_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

            self.n_total = self._positions.npart_total
            self.n_local = self._n()
            self._last_n = self._n()


            if self._return_code[0] < 0:
                if runtime.VERBOSE.level > 2:
                    print "rank:", mpi.MPI_HANDLE.rank, "neighbour list resizing", "old", self.max_len[0], "new", 2*self.max_len[0]
                self.max_len[0] *= 2
                self.list.realloc(self.max_len[0])

                assert _attempt < 20, "Tried to create neighbour list too many times."

                self.update(_attempt + 1)

            self.version_id += 1
            self.update_required = False
            self.max_traveled_distance = 0


neighbour_list = NeighbourList()
"""Defaul/test neighbour list"""










