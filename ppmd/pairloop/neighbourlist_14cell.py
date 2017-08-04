from __future__ import print_function, division

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level imports
import ctypes as ct
import math, os

# package level imports
from ppmd import host, runtime, kernel
import ppmd.lib.shared_lib, ppmd.lib.build

_LIB_SOURCES = os.path.join(os.path.dirname(__file__), 'lib/')

################################################################################################################
# NeighbourListv2 definition 14 cell version
################################################################################################################

class NeighbourListv2(object):
    def __init__(self, list=None):

        # timer inits
        self.timer_update = ppmd.opt.Timer()


        self._cell_list_func = list
        self.cell_list = list
        self.max_len = None
        self.list = None
        self.lib = None

        self.domain_id = 0
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


    def update(self, _attempt=1):

        assert self.max_len is not None and self.list is not None and self._neighbour_lib is not None, "Neighbourlist setup not ran, or failed."

        self.timer_update.start()


        if self.neighbour_starting_points.ncomp < self._n() + 1:
            # print "resizing"
            self.neighbour_starting_points.realloc(self._n() + 1)
        if runtime.VERBOSE > 3:
            print("rank:", self._domain.comm.Get_rank(), "rebuilding neighbour list")


        _n = self.cell_list.cell_list.end - self._domain.cell_count
        self._neighbour_lib.execute(static_args={'end_ix': ct.c_int(self._n()), 'n': ct.c_int(_n)})

        self.n_total = self._positions.npart_total
        self.n_local = self._n()
        self._last_n = self._n()


        if self._return_code[0] < 0:
            if runtime.VERBOSE > 2:
                print("rank:", self._domain.comm.Get_rank(), "neighbour list resizing", "old", self.max_len[0], "new", 2*self.max_len[0])
            self.max_len[0] *= 2
            self.list.realloc(self.max_len[0])

            assert _attempt < 20, "Tried to create neighbour list too many times."

            self.update(_attempt + 1)
            return

        self.version_id = self.cell_list.version_id

        self.timer_update.pause()

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
        self._neighbour_lib = ppmd.lib.shared_lib.SharedLib(_kernel, _dat_dict)



