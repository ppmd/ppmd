# cell list container
import sys
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

        # Timer inits
        self.timer_sort = runtime.Timer(runtime.TIMER, 0)



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

    def get_setup_parameters(self):
        """
        Get the values/function handles used to setup the cell list.
        :return:
        """

        assert (None in (self._n, self._positions, self._domain)) is False, "get_setup_parameters Error: cell list not setup."
        return self._n, self._positions, self._domain


    def setup(self, n_func, positions, domain):
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

        # setup methods to sort into cells.
        # self._cell_sort_setup()
        return

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
                print "Initalisation failed"
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

        const int C0 = 1 + (int)((P[ix*3]     - _b0)*_icel0);
        const int C1 = 1 + (int)((P[ix*3 + 1] - _b2)*_icel1);
        const int C2 = 1 + (int)((P[ix*3 + 2] - _b4)*_icel2);

        const int val = (C2*CA[1] + C1)*CA[0] + C0;
        
        if ((C0 < 1) || (C0 > (CA[0]-2))) {
            cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C0 " << C0 << endl;
            cout << "B[0] " << B[0] << " B[1] " << B[1] << " Px " << P[ix*3+0] << endl;
        }
        if ((C1 < 1) || (C1 > (CA[1]-2))) {
            cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C1 " << C1 << endl;
            cout << "B[2] " << B[2] << " B[3] " << B[3] << " Py " << P[ix*3+1] << endl;
        }
        if ((C2 < 1) || (C2 > (CA[2]-2))) {
            cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C2 " << C2 << endl;
            cout << "B[4] " << B[4] << " B[5] " << B[5] << " Pz " << P[ix*3+2] << endl;
        }

        //printf("ix %d c0 %d c1 %d c2 %d val %d \\n", ix, C0, C1, C2, val);


        //needed, may improve halo exchange times
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
        self._cell_sort_lib = build.SharedLib(_cell_sort_kernel, _dat_dict)

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

            self.timer_sort.start()

            _n = self._cell_list.end - self._domain.cell_count

            self._cell_list[self._cell_list.end] = _n
            self._cell_list.data[_n:self._cell_list.end:] = ct.c_int(-1)
            self._cell_contents_count.zero()

            self._cell_sort_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

            self.version_id += 1
            self.update_required = False

            self.timer_sort.pause()

        else:
            print "CELL LIST NOT INITIALISED"



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
        self._halo_cell_sort_loop = build.SharedLib(_cell_sort_kernel, _cell_sort_dict)


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






# default cell list
#cell_list = CellList()

#print "default cell list is", cell_list

def reset_cell_list():
    global cell_list
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

        self._cell_list_new.data[cell_list.cell_list[cell_list.cell_list.end]:cell_list.cell_list.end:] = ct.c_int(-1)

        self._group_by_cell_lib.execute(static_args={'n':ct.c_int(cell_list.cell_list[cell_list.cell_list.end])})

        # swap array pointers

        self.swaptimer.start()

        # swap dynamic dats
        for idx, ix in enumerate(self._state.particle_dats):
            _tmp = getattr(self._state, ix).data
            getattr(self._state, ix).data = self._new_particle_dats[idx].data
            self._new_particle_dats[idx].data = _tmp


        # swap cell list.
        _tmp = cell_list.cell_list.data
        cell_list.cell_list.data = self._cell_list_new.data
        self._cell_list_new.data = _tmp
        cell_list.cell_list.data[cell_list.cell_list.end] = cell_list.cell_list.end - cell_list.domain.cell_count

        self.swaptimer.pause()

group_by_cell = GroupByCell()


################################################################################################################
# NeighbourList definition 14 cell version
################################################################################################################


class NeighbourList(object):

    def __init__(self, list=None):

        # timer inits
        self.timer_update = runtime.Timer(runtime.TIMER, 0)

        self.cell_list = list
        self.max_len = None
        self.list = None
        self.lib = None

        self.version_id = 0
        """Update tracking of neighbour list. """

        self.cell_width = None
        self.time = 0
        self._time_func = None


        self._positions = None
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


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_int)

        _n = n()
        if _n < 10:
            _n = 10

        _initial_factor = math.ceil(15. * (_n ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))

        if _initial_factor < 100:
            _initial_factor = 100


        self.max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_int)

        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1

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

        const double _b0 = B[0];
        const double _b2 = B[2];
        const double _b4 = B[4];

        const double _icel0 = 1.0/CEL[0];
        const double _icel1 = 1.0/CEL[1];
        const double _icel2 = 1.0/CEL[2];

        const int _ca0 = CA[0];
        const int _ca1 = CA[1];
        const int _ca2 = CA[2];



        // loop over particles
        int m = -1;
        for (int ix=0; ix<end_ix; ix++) {

            const double pi0 = P[ix*3];
            const double pi1 = P[ix*3 + 1];
            const double pi2 = P[ix*3 + 2];

            const int C0 = 1 + (int)((pi0 - _b0)*_icel0);
            const int C1 = 1 + (int)((pi1 - _b2)*_icel1);
            const int C2 = 1 + (int)((pi2 - _b4)*_icel2);

            const int val = (C2*_ca1 + C1)*_ca0 + C0;

            NEIGHBOUR_STARTS[ix] = m + 1;

            for(int k = 0; k < 27; k++){

                int iy = q[n + val + tmp_offset[k]];
                while (iy > -1) {
                    if (ix < iy){

                        const double rj0 = P[iy*3]   - pi0;
                        const double rj1 = P[iy*3+1] - pi1;
                        const double rj2 = P[iy*3+2] - pi2;

                        if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {

                            m++;
                            if (m >= max_len){
                                RC[0] = -1;
                                return;
                            }

                            NEIGHBOUR_LIST[m] = iy;
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
        _dat_dict = {'B': self._domain.boundary,              # Inner boundary on local domain (inc halo cells)
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


    def update(self, _attempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        self.timer_update.start()

        # print _attempt, mpi.MPI_HANDLE.rank, self.timer_update.time()

        if self.neighbour_starting_points.ncomp < self._n() + 1:
            # print "resizing"
            self.neighbour_starting_points.realloc(self._n() + 1)
        if runtime.VERBOSE.level > 3:
            print "rank:", mpi.MPI_HANDLE.rank, "rebuilding neighbour list"


        _n = self.cell_list.cell_list.end - self._domain.cell_count
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

        self.version_id = self.cell_list.version_id

        self.timer_update.pause()

################################################################################################################
# NeighbourListv2 definition 14 cell version
################################################################################################################


class NeighbourListv2(NeighbourList):


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)

        assert self.cell_list.cell_list is not None, "No cell to particle map" \
                                                     " setup"

        #if self.cell_list.cell_list is None:
        #    self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_long)

        _n = n()
        if _n < 10:
            _n = 10

        _initial_factor = math.ceil(15. * (_n ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))

        if _initial_factor < 10:
            _initial_factor = 10


        # print "initial_factor", _initial_factor, 15., (_n**2), domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]

        self.max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_long)


        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1

        # //#define RK %(_RK)s
        _code = '''

        //#define PP ((RK==49) || (RK==50) || (RK==32))
        #define PP (1)

        //cout << "------------------------------" << endl;
        //printf("start P[0] = %%f \\n", P[0]);
        


        const double cutoff = CUTOFF[0];
        const long max_len = MAX_LEN[0];

        const int _h_map[14][3] = {
                                            { 0 , 0 , 0 },
                                            { 0 , 1 , 0 },
                                            { 1 , 0 , 0 },
                                            { 1 , 1 , 0 },
                                            { 1 ,-1 , 0 },

                                            {-1 , 0 , 1 },
                                            {-1 , 1 , 1 },
                                            {-1 ,-1 , 1 },
                                            { 0 , 0 , 1 },
                                            { 0 , 1 , 1 },
                                            { 0 ,-1 , 1 },
                                            { 1 , 0 , 1 },
                                            { 1 , 1 , 1 },
                                            { 1 ,-1 , 1 }
                                        };

        int tmp_offset[14];

        for(int ix=0; ix<14; ix++){
            tmp_offset[ix] = _h_map[ix][0] +
                             _h_map[ix][1] * CA[0] +
                             _h_map[ix][2] * CA[0]* CA[1];
        }

        const int _s_h_map[13][3] = {
                                            {-1 ,-1 ,-1 },
                                            { 0 ,-1 ,-1 },
                                            { 1 ,-1 ,-1 },
                                            {-1 , 0 ,-1 },
                                            { 0 , 0 ,-1 },
                                            { 1 , 0 ,-1 },
                                            {-1 , 1 ,-1 },
                                            { 0 , 1 ,-1 },
                                            { 1 , 1 ,-1 },

                                            {-1 ,-1 , 0 },
                                            { 0 ,-1 , 0 },
                                            {-1 , 0 , 0 },
                                            {-1 , 1 , 0 }
                                        };

        int selective_lookup[13];
        int s_tmp_offset[13];

        for( int ix = 0; ix < 13; ix++){
            selective_lookup[ix] = pow(2, ix);

            s_tmp_offset[ix] = _s_h_map[ix][0] +
                               _s_h_map[ix][1] * CA[0] +
                               _s_h_map[ix][2] * CA[0]* CA[1];
        }


        const double _b0 = B[0];
        const double _b2 = B[2];
        const double _b4 = B[4];
        
        //cout << "boundary" << endl;
        //cout << B[0] << " " << B[1] << endl;
        //cout << B[2] << " " << B[3] << endl;
        //cout << B[4] << " " << B[5] << endl;


        const double _icel0 = 1.0/CEL[0];
        const double _icel1 = 1.0/CEL[1];
        const double _icel2 = 1.0/CEL[2];

        const int _ca0 = CA[0];
        const int _ca1 = CA[1];
        const int _ca2 = CA[2];


        // loop over particles
        long m = -1;
        for (int ix=0; ix<end_ix; ix++) {

            const double pi0 = P[ix*3];
            const double pi1 = P[ix*3 + 1];
            const double pi2 = P[ix*3 + 2];

            const int val = CRL[ix];

            const int C0 = val %% _ca0;
            const int C1 = ((val - C0) / _ca0) %% _ca1;
            const int C2 = (((val - C0) / _ca0) - C1 ) / _ca1;
            if (val != ((C2*_ca1 + C1)*_ca0 + C0) ) {cout << "CELL FAILURE, val=" << val << " 0 " << C0 << " 1 " << C1 << " 2 " << C2 << endl;}


            //cout << "val = " << val << " C0 = " << C0 << " C1 = " << C1 << " C2 = " << C2 << endl;
            //cout << " Ca0 = " << _ca0 << " Ca1 = " << _ca1 << " Ca2 = " << _ca2 << endl;


            NEIGHBOUR_STARTS[ix] = m + 1;

            // non standard directions
            // selective stencil lookup into halo

            int flag = 0;
            if ( C0 == 1 ) { flag |= 6729; }
            if ( C0 == (_ca0 - 2) ) { flag |= 292; }
            if ( C1 == 1 ) { flag |= 1543; }
            if ( C1 == (_ca1 - 2) ) { flag |= 4544; }
            if ( C2 == 1 ) { flag |= 511; }

            // if flag > 0 then we are near a halo
            // that needs attention

            //cout << "flag " << flag << endl;

            if (flag > 0) {

                //check the possble 13 directions
                for( int csx = 0; csx < 13; csx++){
                    if (flag & selective_lookup[csx]){
                        
                        //cout << "S look " << csx << endl;

                        int iy = q[n + val + s_tmp_offset[csx]];
                        while(iy > -1){

                            const double rj0 = P[iy*3]   - pi0;
                            const double rj1 = P[iy*3+1] - pi1;
                            const double rj2 = P[iy*3+2] - pi2;

                            //cout << "S_iy = " << iy << " py0 = " << P[iy*3+0] << " py1 = " << P[iy*3+1] << " py2 = " << P[iy*3+2] << endl;


                            if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {

                                m++;
                                if (m >= max_len){
                                    RC[0] = -1;
                                    return;
                                }

                                NEIGHBOUR_LIST[m] = iy;
                            }

                        iy=q[iy]; }
                    }
                }

                //printf(" ##\\n");

            }

            // standard directions

            for(int k = 0; k < 14; k++){
                
                //cout << "\\toffset: " << k << endl;

                int iy = q[n + val + tmp_offset[k]];
                while (iy > -1) {

                    if ( (tmp_offset[k] != 0) || (iy > ix) ){

                        //if (k==12){ cout << "iy=" << iy << endl;}

                        const double rj0 = P[iy*3]   - pi0;
                        const double rj1 = P[iy*3+1] - pi1;
                        const double rj2 = P[iy*3+2] - pi2;
                        
                        //if (k==12){ cout << "iy=" << iy << " y= " << P[iy*3+0] << " y= " << P[iy*3+1] << " y=" << P[iy*3+2] << endl;}


                        if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {

                            m++;
                            if (m >= max_len){
                                RC[0] = -1;
                                return;
                            }

                            NEIGHBOUR_LIST[m] = iy;
                        }

                    }

                iy = q[iy]; }

            }
        }
        NEIGHBOUR_STARTS[end_ix] = m + 1;

        RC[0] = 0;


        //printf("end P[0] = %%f \\n", P[0]);

        //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        return;
        ''' % {'NULL': ''}



        _dat_dict = {'B': self._domain.boundary,              # Inner boundary on local domain (inc halo cells)
                     'P': self._positions,                    # positions
                     'CEL': self._domain.cell_edge_lengths,   # cell edge lengths
                     'CA': self._domain.cell_array,           # local domain cell array
                     'q': self.cell_list.cell_list,           # cell list
                     'CRL': self.cell_list.cell_reverse_lookup,
                     'CUTOFF': self.cell_width_squared,
                     'NEIGHBOUR_STARTS': self.neighbour_starting_points,
                     'NEIGHBOUR_LIST': self.list,
                     'MAX_LEN': self.max_len,
                     'RC': self._return_code}

        _static_args = {'end_ix': ct.c_int,  # Number of particles.
                        'n': ct.c_int}       # start of cell point in list.


        _kernel = kernel.Kernel('neighbour_list_v2', _code, headers=['stdio.h'], static_args=_static_args)
        self._neighbour_lib = build.SharedLib(_kernel, _dat_dict)




################################################################################################################
# NeighbourMatrix definition 14 cell version experimental
################################################################################################################


class NeighbourMatrix(object):

    def __init__(self, list=None):

        # timer inits
        self.timer_update = runtime.Timer(runtime.TIMER, 0)

        self.cell_list = cell_list
        self.max_len = None
        self.list = None
        self.lib = None

        self.version_id = 0
        """Update tracking of neighbour list. """

        self.cell_width = None
        self.time = 0
        self._time_func = None


        self._positions = None
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


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_int)

        _initial_factor = math.ceil(15. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))


        self.max_len = host.Array(initial_value=_initial_factor/n(), dtype=ct.c_int)

        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1

        _code = '''
        
        //printf("starting \\n");

        const double cutoff = CUTOFF[0];
        const int max_len = MAX_LEN[0];

        const int _h_map[14][3] = {
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


        const double _b0 = B[0];
        const double _b2 = B[2];
        const double _b4 = B[4];

        const double _icel0 = 1.0/CEL[0];
        const double _icel1 = 1.0/CEL[1];
        const double _icel2 = 1.0/CEL[2];

        const int _ca0 = CA[0];
        const int _ca1 = CA[1];
        const int _ca2 = CA[2];
        
        // reset numbers of neighbours
        for (int ix=0; ix<end_ix; ix++) {
            NEIGHBOUR_STARTS[ix] = 0;
        }

        //printf("reset counts \\n");

        //loop over cells
        for(int _cx = 0; _cx < _ca0; _cx++){ 
            
            //printf("_cx=%d c0=%d\\n", _cx, _ca0);

            for( int _cy = 0; _cy < _ca1; _cy++){
                
                //printf("\\t _cy=%d c1=%d\\n", _cy, _ca1);

                for(int _cz = 0; _cz < _ca2-1; _cz++){
                    
                    //printf("\\t \\t _cz=%d c2-1=%d\\n", _cz, _ca2-1);

                    int _cp_halo_flag = 0;

                    //we are now sat in some cell
                    const int _cp = _cz*(_ca0*_ca1) + _cy*(_ca0) + _cx;

                    if ( 
                        (_cx == 0) || (_cx == (_ca0-1)) ||
                        (_cy == 0) || (_cy == (_ca1-1)) ||
                        (_cz == 0)
                    ) { _cp_halo_flag = 1; }
                    
                    for(int _cpp_i = 0; _cpp_i < 14; _cpp_i++){

                        const int _cpp_x = (_cx + _h_map[_cpp_i][0]) % _ca0;
                        const int _cpp_y = (_cy + _h_map[_cpp_i][1]) % _ca1;
                        const int _cpp_z = (_cz + _h_map[_cpp_i][2]) % _ca2;
                        
                        int _cpp_halo_flag = 0;

                        if ( 
                            (_cpp_x == 0) || (_cpp_x == (_ca0-1)) ||
                            (_cpp_y == 0) || (_cpp_y == (_ca1-1)) ||
                            (_cpp_z == 0) || (_cpp_z == (_ca2 -1))
                        ) { _cpp_halo_flag = 1; }
                        
                        //are both cells in a halo?
                        if (_cp_halo_flag && _cpp_halo_flag) {break;}
                        
                        const int _cpp = _cpp_z*(_ca0*_ca1) + _cpp_y*(_ca0) + _cpp_x;



                        int ix = q[n + _cp];
                        while (ix > -1) {
                            
                            const double pi0 = P[ix*3];
                            const double pi1 = P[ix*3 + 1];
                            const double pi2 = P[ix*3 + 2];

                            int iy = q[n + _cpp];
                            while (iy > -1) {
                                
                                if((_cp != _cpp) || (ix < iy)){

                                const double rj0 = P[iy*3]   - pi0;
                                const double rj1 = P[iy*3+1] - pi1;
                                const double rj2 = P[iy*3+2] - pi2;

                                if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff ) {
                                    
                                    if (_cp_halo_flag < 1){
                                        const int _tix = NEIGHBOUR_STARTS[ix];

                                        if(_tix >= max_len){
                                            RC[0] = -1;
                                            //printf("finish unintended\\n");
                                            return;
                                        }

                                        NEIGHBOUR_LIST[max_len*ix + _tix] = iy;
                                        NEIGHBOUR_LIST[ix]++;
                                    }
                                    
                                    if(_cpp_halo_flag < 1){

                                        const int _tiy = NEIGHBOUR_STARTS[iy];

                                        if(_tiy >= max_len) {
                                            RC[0] = -1;
                                            //printf("finish unintended\\n");
                                            return;
                                        }

                                        NEIGHBOUR_LIST[max_len*iy + _tiy] = ix;
                                        NEIGHBOUR_LIST[iy]++;
                                        }

                                    }

   

                                }

                            iy = q[iy];
                            }

                        ix = q[ix];
                        }

                    }
        
        }}}


        RC[0] = 0;

        //printf("finish intented \\n");

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


        _kernel = kernel.Kernel('cell_neighbour_matrix_method', _code, headers=['stdio.h'], static_args=_static_args)
        self._neighbour_lib = build.SharedLib(_kernel, _dat_dict)


    def update(self, _attempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        self.timer_update.start()

        if self.neighbour_starting_points.ncomp < self._n() + 1:
            self.neighbour_starting_points.realloc(self._n() + 1)
            self.list.realloc(self.max_len[0] * self._n())

        if runtime.VERBOSE.level > 3:
            print "rank:", mpi.MPI_HANDLE.rank, "rebuilding neighbour list"


        _n = self.cell_list.cell_list.end - self._domain.cell_count
        self._neighbour_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

        self.n_total = self._positions.npart_total
        self.n_local = self._n()
        self._last_n = self._n()


        if self._return_code[0] < 0:
            if runtime.VERBOSE.level > 2:
                print "rank:", mpi.MPI_HANDLE.rank, "neighbour list resizing", "old", self.max_len[0], "new", 2*self.max_len[0]
            self.max_len[0] *= 2
            self.list.realloc(self.max_len[0]*self._n())

            assert _attempt < 20, "Tried to create neighbour list too many times."

            self.update(_attempt + 1)

        self.version_id += 1

        self.timer_update.pause()










################################################################################################################
# NeighbourList with inner and outer lists.
################################################################################################################


class NeighbourListHaloAware(object):

    def __init__(self, list=None):
        self.cell_list = cell_list
        self.max_len = None
        self.list = None
        self.lib = None

        self.version_id = 0
        """Update tracking of neighbour list. """

        self.cell_width = None
        self.time = 0
        self._time_func = None


        self._positions = None
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

        #halo neighbourlist
        self._halo_neighbour_lib = None
        self.halo_neighbour_starting_points = None
        self.halo_max_len = None
        self.halo_list = None
        self._boundary_cells = None
        self.halo_part_count = None


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_int)

        _initial_factor = math.ceil(15. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))


        self.max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_int)

        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1




        _code = '''

        const double cutoff = CUTOFF[0];
        const int max_len = MAX_LEN[0];

        const int _h_map[14][3] = {         {0,0,0},
                                            {1,0,0},
                                            {0,1,0},
                                            {1,1,0},
                                            {1,-1,0},
                                            {-1,1,1},
                                            {0,1,1},
                                            {1,1,1},
                                            {-1,0,1},
                                            {0,0,1},
                                            {1,0,1},
                                            {-1,-1,1},
                                            {0,-1,1},
                                            {1,-1,1}};

        int tmp_offset[14];
        for(int ix=0; ix<14; ix++){
            tmp_offset[ix] = _h_map[ix][0] + _h_map[ix][1] * CA[0] + _h_map[ix][2] * CA[0]* CA[1];
        }

        // loop over particles
        int m = -1;
        for (int ix=0; ix<end_ix; ix++) {


            const int C0 = 1 + (int)((P[ix*3]     - B[0])/CEL[0]);
            const int C1 = 1 + (int)((P[ix*3 + 1] - B[2])/CEL[1]);
            const int C2 = 1 + (int)((P[ix*3 + 2] - B[4])/CEL[2]);

            const int val = (C2*CA[1] + C1)*CA[0] + C0;

            NEIGHBOUR_STARTS[ix] = m + 1;

            for(int k = 0; k < 14; k++){

                int iy = q[n + val + tmp_offset[k]];
                while ((iy > -1) && (iy < end_ix)) {
                    if ((tmp_offset[k] != 0) || (ix < iy)){

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
        _dat_dict = {'B': self._domain.boundary,              # Inner boundary on local domain (inc halo cells)
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


        #----------------- halo lib ----------------------------


        self.halo_neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_int)

        _initial_factor = math.ceil(15. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))


        self.halo_max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_int)

        self.halo_list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)

        self._boundary_cells = self._domain.get_boundary_cells()

        self.halo_part_count = host.Array(ncomp=1, dtype=ct.c_int)


        _code = '''

        const double cutoff = CUTOFF[0];
        const int max_len = MAX_LEN[0];

        const int _h_map[26][3] = {
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

        int tmp_offset[26];
        for(int ix=0; ix<26; ix++){
            tmp_offset[ix] = _h_map[ix][0] + _h_map[ix][1] * CA[0] + _h_map[ix][2] * CA[0]* CA[1];
        }

        // loop over boundary cells
        int m = -1;
        int ns = -2;

        //printf("end_ix=%d \\n", end_ix);



        for (int cx=0; cx<end_ix; cx++) {

            const int val = B_cells[cx];


            //printf("val=%d \\n", val);

            int ix = q[n + val];
            while (ix > -1){

                ns += 2;


                //printf("ns=%d, m=%d, ix=%d \\n", ns, m, ix);
                NEIGHBOUR_STARTS[ns] = m + 1;
                NEIGHBOUR_STARTS[ns + 1] = ix;


                for(int k = 0; k < 26; k++){

                    int iy = q[n + val + tmp_offset[k]];
                    while (iy >= N_LOCAL) {

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

                        iy = q[iy];
                    }

                }

                ix = q[ix];
            }

        }
        ns += 2;
        NEIGHBOUR_STARTS[ns] = m + 1;
        if (ns > 0){
            HPC[0] = ns/2;
        } else {
            HPC[0] = 0;
        }

        RC[0] = 0;
        return;
        '''
        _halo_dat_dict = {'B_cells': self._boundary_cells,         # boundary cells
                          'B': self._domain.boundary_outer,        # Outer boundary on local domain (inc halo cells)
                          'P': self._positions,                    # positions
                          'CEL': self._domain.cell_edge_lengths,   # cell edge lengths
                          'CA': self._domain.cell_array,           # local domain cell array
                          'q': self.cell_list.cell_list,           # cell list
                          'CUTOFF': self.cell_width_squared,
                          'NEIGHBOUR_STARTS': self.halo_neighbour_starting_points,
                          'NEIGHBOUR_LIST': self.halo_list,
                          'MAX_LEN': self.halo_max_len,
                          'RC': self._return_code,
                          'HPC': self.halo_part_count}

        _halo_static_args = {'end_ix': ct.c_int,  # Number of boundary cells.
                             'n': ct.c_int,       # start of cell point in list.
                             'N_LOCAL': ct.c_int  # local number of particles.
                             }

        _halo_kernel = kernel.Kernel('cell_neighbour_list_method_halo', _code, headers=['stdio.h'], static_args=_halo_static_args)
        self._halo_neighbour_lib = build.SharedLib(_halo_kernel, _halo_dat_dict)
















    def update(self, _attempt=1, _hattempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        if self.neighbour_starting_points.ncomp < self._n() + 1:
            self.neighbour_starting_points.realloc(self._n() + 1)
            self.halo_neighbour_starting_points.realloc(self._n() + 1)

        if runtime.VERBOSE.level > 3:
            print "rank:", mpi.MPI_HANDLE.rank, "rebuilding neighbour list"


        _n = self.cell_list.cell_list.end - self._domain.cell_count
        self._neighbour_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

        self.n_total = self._positions.npart_total
        self.n_local = self._n()
        self._last_n = self._n()


        if self._return_code[0] < 0:
            if runtime.VERBOSE.level > 2:
                print "rank:", mpi.MPI_HANDLE.rank, "neighbour list resizing", "old", self.max_len[0], "new", 2 * self.max_len[0]
            self.max_len[0] *= 2
            self.list.realloc(self.max_len[0])

            assert _attempt < 20, "Tried to create neighbour list too many times."

            self.update(_attempt=_attempt + 1)


        self.version_id += 1

    def halo_update(self, _hattempt=1):

        _n = self.cell_list.cell_list.end - self._domain.cell_count

        self._boundary_cells = self._domain.get_boundary_cells()

        # print self._boundary_cells.ncomp

        self._halo_neighbour_lib.execute(static_args={'end_ix': ct.c_int(self._boundary_cells.ncomp), 'n': ct.c_int(_n), 'N_LOCAL': ct.c_int(self._n())})
        if self._return_code[0] < 0:
            if runtime.VERBOSE.level > 2:
                print "rank:", mpi.MPI_HANDLE.rank, "halo neighbour list resizing", "old", self.halo_max_len[0], "new", 2 * self.halo_max_len[0]
            self.halo_max_len[0] *= 2
            self.halo_list.realloc(self.halo_max_len[0])

            assert _hattempt < 20, "Tried to create neighbour list too many times."

            self.update(_hattempt=_hattempt + 1)

















################################################################################################################
# NeighbourList definition 27 cell version
################################################################################################################

class NeighbourListNonN3(NeighbourList):


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = host.Array(initial_value=cell_width ** 2, dtype=ct.c_double)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."

        self.neighbour_starting_points = host.Array(ncomp=n() + 1, dtype=ct.c_int)

        _initial_factor = math.ceil(27. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2]))


        self.max_len = host.Array(initial_value=_initial_factor, dtype=ct.c_int)

        self.list = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1

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

            const int C0 = 1 + (int)((P[ix*3]     - B[0])/CEL[0]);
            const int C1 = 1 + (int)((P[ix*3 + 1] - B[2])/CEL[1]);
            const int C2 = 1 + (int)((P[ix*3 + 2] - B[4])/CEL[2]);

            const int val = (C2*CA[1] + C1)*CA[0] + C0;

            NEIGHBOUR_STARTS[ix] = m + 1;

            for(int k = 0; k < 27; k++){

                int iy = q[n + val + tmp_offset[k]];
                while (iy > -1) {
                    if (ix != iy){

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
        _dat_dict = {'B': self._domain.boundary,              # Inner boundary on local domain (inc halo cells)
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


    def update(self, _attempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        if self.neighbour_starting_points.ncomp < self._n() + 1:
            self.neighbour_starting_points.realloc(self._n() + 1)
        if runtime.VERBOSE.level > 3:
            print "rank:", mpi.MPI_HANDLE.rank, "rebuilding neighbour list"


        _n = cell_list.cell_list.end - self._domain.cell_count
        self._neighbour_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

        self.n_total = self._positions.npart_total
        self.n_local = self._n()
        self._last_n = self._n()


        if self._return_code[0] < 0:
            if runtime.VERBOSE.level > 2:
                print "rank:", mpi.MPI_HANDLE.rank, "neighbour list resizing", "old", self.max_len[0], "new", 2 * self.max_len[0]
            self.max_len[0] *= 2
            self.list.realloc(self.max_len[0])

            assert _attempt < 20, "Tried to create neighbour list too many times."

            self.update(_attempt + 1)

        self.version_id += 1


################################################################################################################
# CPU CelllayerSort
################################################################################################################


class CellLayerSort(object):
    def __init__(self):
        self._cell_list = None
        self._lib = None
        self._lib2 = None
        self.cell_occupancy_matrix = None
        """Get the cell occupancy matrix."""

        self.particle_layers = None
        """Get the particle layer index."""

        self.num_layers = None
        """Maximum number of layers in use."""

        self.version_id = 0
        """version id for cell layer"""

    def cell_occupancy_counter(self):
        """Array containing the number of atoms per cell"""
        assert self._cell_list is not None, "CellLayerSort error: run setup first."
        return self._cell_list.cell_contents_count

    def setup(self, list_in=None, openmp=False):
        assert list_in is not None, "CellLayerSort setup error: no CellList passed."
        self._cell_list = list_in
        self.particle_layers = host.Array(ncomp=self._cell_list.num_particles, dtype=ct.c_int)
        self.cell_occupancy_matrix = host.Array(ncomp=self._cell_list.domain.cell_count, dtype=ct.c_int)


        _code = '''

            #ifdef _OPENMP
            #pragma omp for
            #endif
            for(int cx = 0; cx < Nc; cx++){

                int l = 0;
                int ix = CELL_LIST[CL_start + cx];
                while(ix > -1){
                    L[ix] = l;
                    l++;
                    ix = CELL_LIST[ix];
                }
            }

        '''

        _statics = {
            'Nc': ct.c_int,  # Num_cells
            'CL_start': ct.c_int  # starting point of cells in array
        }

        _dynamics = {
            'CELL_LIST': self._cell_list.cell_list,
            'L': self.particle_layers
        }

        _kernel = kernel.Kernel('layers_sort_method', _code, headers=['stdio.h'], static_args=_statics)

        self._lib = build.SharedLib(_kernel, _dynamics, openmp)

        _code2 = '''


            for(int cx = 0; cx < Nc; cx++){
                H[cx * (Lm+1)] = COC[cx];
                //printf("COC[cx]=%d \\n", COC[cx]);
            }

            #ifdef _OPENMP
            #pragma omp for
            #endif
            for(int ix = 0; ix < Na; ix++){

                H[ CRL[ix]*(Lm+1) + L[ix] + 1 ] = ix;

            }

        '''

        _statics2 = {
            'Na': ct.c_int,  # Number of atoms
            'Nc': ct.c_int,  # Number of cells
            'Lm': ct.c_int  # Max number of layers
        }

        _dynamics2 = {
            'CRL': self._cell_list.cell_reverse_lookup,
            'L': self.particle_layers,
            'H': self.cell_occupancy_matrix,
            'COC': self._cell_list.cell_contents_count
        }

        _kernel2 = kernel.Kernel('cell__layer_occupancy', _code2, headers=['stdio.h'], static_args=_statics2)
        self._lib2 = build.SharedLib(_kernel2, _dynamics2, openmp)

    def update(self):
        assert self._lib is not None, "CellLayerSort error: setup not ran or failed."
        assert self._lib2 is not None, "CellLayerSort error: setup not ran or failed."

        _Nc = self._cell_list.domain.cell_count
        _CL_start = ct.c_int(self._cell_list.cell_list[self._cell_list.cell_list.end])

        _Nc = ct.c_int(_Nc)


        if self.particle_layers.ncomp < self._cell_list.total_num_particles:
            self.particle_layers.realloc(self._cell_list.total_num_particles)


        self._lib.execute(static_args={'Nc': _Nc, 'CL_start': _CL_start})

        _Lm = self._cell_list.cell_contents_count.data[0:self._cell_list.domain.cell_count:].max()

        self.num_layers = _Lm

        _Na = ct.c_int(self._cell_list.total_num_particles)

        if self.cell_occupancy_matrix.ncomp < (_Lm + 1) * self._cell_list.domain.cell_count:
            self.cell_occupancy_matrix.realloc((_Lm + 1) * self._cell_list.domain.cell_count)
        _Lm = ct.c_int(_Lm)

        _statics2 = {'Na': _Na, 'Nc': _Nc, 'Lm': _Lm}
        self._lib2.execute(static_args=_statics2)
        self.version_id += 1

        #print "occupancy matrix", _Lm, self.cell_occupancy_matrix.data
        #print "particle layers", self.particle_layers.data





################################################################################################################
# CPU NeighbourListLayerBased
################################################################################################################


class NeighbourListLayerBased(object):
    def __init__(self):
        self.list = None
        self._lmi = None
        self._cli = None
        self._lib = None
        self._cutoff_squared = None

        self.version_id = 0
        """version id for neighbour matrix"""

        self.neighbour_matrix = None
        """Neighbour matrix (host.Array)"""

        self.num_neighbours_per_atom = 0
        """maxmium_number of neighbours per atom"""

    def setup(self, layer_method_instance, cell_list_instance, openmp=False):
        self._lmi = layer_method_instance
        self._cli = cell_list_instance

        self.neighbour_matrix = host.Array(ncomp=self._cli.num_particles * 2, dtype=ct.c_int)
        self._cutoff_squared = host.Array(ncomp=1, dtype=ct.c_double)
        self._cutoff_squared[0] = self._cli.cell_width ** 2

        _code = '''
        const double cutoff = CUTOFF[0];
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

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for(int ix = 0; ix < Na; ix++){ // Loop over particles.

            const double r00 = P[ix*3];
            const double r01 = P[ix*3+1];
            const double r02 = P[ix*3+2];

            int m = 0; // index of next neighbour. Use 0 for the number of neighbours.
            const int cp = CRL[ix]; //cell index containing this particle.

            //printf("ix=%d, cp=%d \\n", ix, cp);

            for (int k = 0; k < 27; k++) { // Loop over cell directions.
                const int cpp = cp + tmp_offset[k];

                for(int _iy = 1; _iy < H[cpp*(Lm+1)]+1; _iy++){ //traverse layers in cell cpp.

                    int iy = H[cpp*(Lm+1) + _iy];

                    //printf("ix=%d iy=%d \\n", ix, iy);


                    if (ix != iy ){
                        const double r10 = P[iy*3]   - r00;
                        const double r11 = P[iy*3+1] - r01;
                        const double r12 = P[iy*3+2] - r02;

                        if ( (r10*r10 + r11*r11 + r12*r12) < cutoff ){

                            m++;
                            if(m >= Nn+1) {printf("Error Maximum number of neighbours reached(%d) \\n", Nn+1);}

                            W[(ix*(Nn+1)) + m] = iy; //Putting neighbours of the same atom contiguous in
                                                 //memory, the opposite to the gpu approach. This
                                                 //should be checked.

                        }

                    }

                }
            }

            W[ix*(Nn+1)] = m; // Records the number of neighbours.
            //printf("NSTAGE i=%d, m=%d \\n", ix, m);
        }

        '''

        _statics = {
            'Na': ct.c_int,  # Na number of atoms.
            'Lm': ct.c_int,  # Lm: Number of layers to use for indexing for cell occupancy matrix.
            'Nn': ct.c_int   # Nn: Maximum number of neighbours per atom accounting for the +1.
        }

        _dynamics = {
            'CRL': self._cli.cell_reverse_lookup,
            'CA': self._cli.domain.cell_array,
            'W': self.neighbour_matrix,
            'H': self._lmi.cell_occupancy_matrix,
            'CUTOFF': self._cutoff_squared,
            'P': self._cli.get_setup_parameters()[1]
        }

        _kernel = kernel.Kernel('neighbour_matrix_creation', _code, headers=['stdio.h'], static_args=_statics)
        self._lib = build.SharedLib(_kernel, _dynamics, openmp)


    def update(self):
        assert self._lib is not None, "NeighbourListLayerBased error: library not created."

        _Nn = 1 + self._lmi.num_layers * 27


        if self.neighbour_matrix.ncomp < _Nn * self._cli.total_num_particles:
                self.neighbour_matrix.realloc(_Nn * self._cli.total_num_particles)

        _Na = ct.c_int(self._cli.num_particles)
        _Lm = ct.c_int(self._lmi.num_layers)

        self.num_neighbours_per_atom = _Nn - 1
        _Nn = ct.c_int(_Nn - 1)
        _statics = {'Na': _Na, 'Lm': _Lm, 'Nn':_Nn}

        self._lib.execute(static_args=_statics)
        self.version_id += 1
        #print "Nn", _Nn
        #print "0", self.neighbour_matrix.data[0:_tnn:]
        #print "1", self.neighbour_matrix.data[_tnn:2*_tnn:]






















################################################################################################################
# PairNeighbourList definition 14 cell version
################################################################################################################


class NeighbourListPairIndices(object):

    def __init__(self, list=None):
        self.cell_list = cell_list
        self.max_len = None

        self.listi = None
        self.listj = None
        self.list_length = ct.c_int(0)


        self.version_id = 0
        """Update tracking of neighbour list. """

        self.cell_width = None
        self.time = 0
        self._time_func = None


        self._positions = None
        self._domain = None
        self.cell_width_squared = None
        self._neighbour_lib = None
        self._n = None

        self.n_local = None


        self._return_code = None


    def setup(self, n, positions, domain, cell_width):

        # setup the cell list if not done already (also handles domain decomp)
        if self.cell_list.cell_list is None:
            self.cell_list.setup(n, positions, domain, cell_width)

        self.cell_width = cell_width

        self.cell_width_squared = ct.c_double(cell_width ** 2)
        self._domain = domain
        self._positions = positions
        self._n = n

        # assert self._domain.halos is True, "Neighbour list error: Only valid for domains with halos."


        _initial_factor = int(math.ceil(15. * (n() ** 2) / (domain.cell_array[0] * domain.cell_array[1] * domain.cell_array[2])))

        self.max_len = ct.c_int(_initial_factor)

        self.listi = host.Array(ncomp=_initial_factor, dtype=ct.c_int)
        self.listj = host.Array(ncomp=_initial_factor, dtype=ct.c_int)


        self._return_code = host.Array(ncomp=1, dtype=ct.c_int)
        self._return_code.data[0] = -1

        _name = 'pairwise_neighbour_list'


        _args = '''
                const int q_na,                     //starting point in cell list
                const int* __restrict__ q,          //cell list.
                const int * __restrict__ CA,        //cell array
                const double* __restrict__ P,       //particle positions
                const double cutoff2,               //cutoff squared
                const int max_len,                  //maximum length of lists.
                int* RC,                            //return code
                int* __restrict__ LIST_I,           //list i
                int* __restrict__ LIST_J,           //list j
                int* PAIR_COUNT                     //number of pairs.
                '''

        _header = '''
        #include <generic.h>
        extern "C" void pairwise_neighbour_list(%(ARGS)s);
        ''' % {'ARGS':_args}

        _code = '''
        void pairwise_neighbour_list(%(ARGS)s){

        const int _h_map[27][3] = {
{-1,1,-1}, {-1,-1,-1}, {-1,0,-1}, {0,1,-1}, {0,-1,-1}, {0,0,-1}, {1,0,-1}, {1,1,-1}, {1,-1,-1},  {-1,1,0}, {-1,0,0}, {-1,-1,0}, {0,-1,0}, {0,0,0}, {0,1,0}, {1,0,0}, {1,1,0}, {1,-1,0},  {-1,0,1}, {-1,1,1}, {-1,-1,1}, {0,0,1}, {0,1,1}, {0,-1,1}, {1,0,1}, {1,1,1}, {1,-1,1}  };


        int tmp_offset[27];
        for(int ix=0; ix<27; ix++){
            tmp_offset[ix] = _h_map[ix][0] + _h_map[ix][1] * CA[0] + _h_map[ix][2] * CA[0]* CA[1];
        }
        
        //current index.
        int m = -1;


        for(int k = 0; k < 27; k++){

            for (int cx = 1; cx < CA[0]-1; cx++){ for (int cy = 1; cy < CA[1]-1; cy++){ for (int cz = 1; cz < CA[2]-1; cz++){
            
                const int bx = cx + CA[0] * (cy + cz*CA[1]);

                int ix = q[q_na + bx];
                while(ix > -1){
                    
                    int iy = q[q_na + bx + tmp_offset[k]];
                    while (iy > -1){
                        
                        if (ix < iy){

                            const double rj0 = P[iy*3]   - P[ix*3];
                            const double rj1 = P[iy*3+1] - P[ix*3+1];
                            const double rj2 = P[iy*3+2] - P[ix*3+2];

                            if ( (rj0*rj0 + rj1*rj1 + rj2*rj2) <= cutoff2 ) {

                                m++;
                                if (m >= max_len){
                                    RC[0] = -1;
                                    return;
                                }

                                LIST_I[m] = ix;
                                LIST_J[m] = iy;
                            }

                        }


                        iy = q[iy];
                    }
        
                    ix = q[ix];
                }

            }}}
        }


        *PAIR_COUNT = m+1;
        RC[0] = 0;
        return;

        }
        ''' % {'ARGS': _args}


        self._neighbour_lib = build.simple_lib_creator(_header, _code, _name)[_name]


    def update(self, _attempt=1):

        assert self.max_len is not None and self.listi is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        
        _initial_factor = int(math.ceil(15. * (self._n() ** 2) / (self._domain.cell_array[0] * self._domain.cell_array[1] * self._domain.cell_array[2])))
        
        if self.listi.ncomp < _initial_factor:
            self.listi.realloc(_initial_factor)
            self.listj.realloc(_initial_factor)
            self.max_len.value = _initial_factor 
        if runtime.VERBOSE.level > 3:
            print "rank:", mpi.MPI_HANDLE.rank, "rebuilding neighbour list"


        _n = self.cell_list.cell_list.end - self._domain.cell_count

        self._neighbour_lib(
                ct.c_int(_n),
                self.cell_list.cell_list.ctypes_data,
                self._domain.cell_array.ctypes_data,
                self._positions.ctypes_data,
                self.cell_width_squared,
                self.max_len,
                self._return_code.ctypes_data,
                self.listi.ctypes_data,
                self.listj.ctypes_data,
                ct.byref(self.list_length)
                )


        '''
        const int q_na,                     //starting point in cell list
        const int* __restrict__ q,          //cell list.
        const int * __restrict__ CA,        //cell array
        const double* __restrict__ P,       //particle positions
        const double cutoff2,               //cutoff squared
        const int max_len,                  //maximum length of lists.
        int* RC,                            //return code
        int* __restrict__ LIST_I,           //list i
        int* __restrict__ LIST_J,           //list j
        int* PAIR_COUNT                     //number of pairs.
        '''


        self.n_total = self._positions.npart_total
        self.n_local = self._n()


        if self._return_code[0] < 0:
            if runtime.VERBOSE.level > 2:
                print "rank:", mpi.MPI_HANDLE.rank, "neighbour list resizing", "old", self.max_len.value, "new", 2*self.max_len.value
            self.max_len.value *= 2
            self.listi.realloc(self.max_len.value)
            self.listj.realloc(self.max_len.value)

            assert _attempt < 20, "Tried to create neighbour list too many times."

            self.update(_attempt + 1)

        self.version_id += 1



