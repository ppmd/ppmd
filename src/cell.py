# cell list container
import runtime
import data
import particle
import ctypes as ct
import numpy as np
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



        return _err

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

    @property
    def domain(self):
        """
        :return: The domain used.
        """
        return self._domain

# default cell list
cell_list = CellList()

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
            self._new_particle_dats.append(particle.Dat(n1=_dat.npart, n2=_dat.ncomp, dtype=_dat.dtype))
            self._sizes.append(_dat.ncomp)

        self._cell_list_new = data.ScalarArray(ncomp=cell_list.cell_list.max_size, dtype=ct.c_int)

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
