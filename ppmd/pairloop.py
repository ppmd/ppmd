# system level
import ctypes
import os
import cgen

# package level
import generation
import data
import build
import runtime
import access
import cell
import host
import opt
import logic

class _Base(object):
    def __init__(self, n, kernel, particle_dat_dict):

        self._cc = build.TMPCC

        self._N = n

        self._kernel = kernel

        self._particle_dat_dict = particle_dat_dict

        self.loop_timer = opt.LoopTimer()

        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)
    def _code_init(self):
        pass

    def _generate_header_source(self):
        pass

    def _argnames(self):
        """Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of
        the method which executes the pairloop over the grid.
        If, for example, the pairloop gets passed two particle_dats,
        then the result will be ``double** arg_000,double** arg_001`.`
        """


        argnames = str(self.loop_timer.get_cpp_arguments()) + ','


        if self._kernel.static_args is not None:
            self._static_arg_order = []

            for i, dat in enumerate(self._kernel.static_args.items()):
                argnames += 'const ' + host.ctypes_map[dat[1]] + ' ' + dat[0] + ','
                self._static_arg_order.append(dat[0])


        for i, dat in enumerate(self._particle_dat_dict.items()):




            if type(dat[1]) is not tuple:
                argnames += host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'
            else:
                if not dat[1][1].write:
                    const_str = 'const '
                else:
                    const_str = ''
                argnames += const_str + host.ctypes_map[dat[1][0].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'


        return argnames[:-1]

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + x + '\" \n'

        s += str(self.loop_timer.get_cpp_headers())

        return s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations(),
             'LOOP_TIMER_PRE': str(self.loop_timer.get_cpp_pre_loop_code()),
             'LOOP_TIMER_POST': str(self.loop_timer.get_cpp_post_loop_code()),
             'RESTRICT': self._cc.restrict_keyword}

        return self._code % d

    def execute(self, n=None, dat_dict=None, static_args=None):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        '''Currently assume n is always needed'''
        if n is not None:
            _N = n
        else:
            _N = self._N()

        args = [ctypes.c_int(_N)]

        args.append(self.loop_timer.get_python_parameters())


        '''TODO IMPLEMENT/CHECK RESISTANCE TO ARG REORDERING'''

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']
        method(*args)

        '''after wards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])



###############################################################################
# RAPAPORT LOOP SERIAL FOR HALO DOMAINS
###############################################################################


class PairLoopRapaportHalo(_Base):
    def __init__(self, domain, potential=None, dat_dict=None, kernel=None):
        self._domain = domain
        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        self.loop_timer = opt.LoopTimer()

        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)

    def _kernel_argument_declarations(self):
        s = build.Code()
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return str(s)

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"
        
        extern "C" void %(KERNEL_NAME)s_wrapper(const int n, int* cell_array, int* q_list,%(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include <stdio.h>
        
        void cell_index_offset(const unsigned int cp_i, const unsigned int cpp_i, int* cell_array, unsigned int* cpp, unsigned int* cp_h_flag, unsigned int* cpp_h_flag){
        
            const int cell_map[14][3] = {   {0,0,0},
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
            
            //internal cell array dimensions
            const unsigned int ca0 = cell_array[0];
            const unsigned int ca1 = cell_array[1];
            const unsigned int ca2 = cell_array[2];
            
            //Get index for cell cp as tuple: cp=(Cx, Cy, Cz)
            const int tmp = ca0*ca1;
            int Cz = cp_i/tmp;
            int Cx = cp_i %% ca0;
            int Cy = (cp_i - Cz*tmp)/ca0;
            
            //check if cp is in halo
            if( (Cx %% (ca0-1) == 0) || (Cy %% (ca1-1) == 0) || (Cz %% (ca2-1) == 0)){
                *cp_h_flag = 1;
            }else{*cp_h_flag = 0;}
            
            
            //compute tuple of adjacent cell cpp
            Cx = (Cx + cell_map[cpp_i][0] + ca0) %% ca0;
            Cy = (Cy + cell_map[cpp_i][1] + ca1) %% ca1;
            Cz = (Cz + cell_map[cpp_i][2] + ca2) %% ca2;
            
            //compute linear index of adjacent cell cpp
            *cpp = Cz*(ca0*ca1) + Cy*(ca0) + Cx;
            
            //check if cpp is in halo
            if( (Cx %% (ca0-1) == 0) || (Cy %% (ca1-1) == 0) || (Cz %% (ca2-1) == 0)){
                *cpp_h_flag = 1;
            }else{*cpp_h_flag = 0;}          
            
            
                
            return;      
        }    
        
        void %(KERNEL_NAME)s_wrapper(const int n, int* cell_array, int* q_list,%(ARGUMENTS)s) { 
            
            //printf("starting");
            for(unsigned int cp = 0; cp < cell_array[0]*cell_array[1]*(cell_array[2]-1); cp++){
                for(unsigned int cpp_i=0; cpp_i<14; cpp_i++){
                    
                    //printf("cp=%%d, cpp_i=%%d |",cp,cpp_i);
                    
                    
                    unsigned int cpp, _cp_halo_flag, _cpp_halo_flag;
                    int _i,_j;
                    
                    
                    cell_index_offset(cp, cpp_i, cell_array, &cpp, &_cp_halo_flag, &_cpp_halo_flag);
                    
                    //Check that both cells are not halo cells.
                    
                    //printf("cpp=%%d, flagi=%%d, flagj=%%d |",cpp, cp_h_flag,cpp_h_flag);
                    
                    if ((_cp_halo_flag+_cpp_halo_flag) < 2){
                        
                        _i = q_list[n+cp];
                        while (_i > -1){
                            _j = q_list[n+cpp];
                            while (_j > -1){
                                if (cp != cpp || _i < _j){
                                    
                                    double *ri, *rj;
                                    
                                    double null_array[3] = {0,0,0};
                                    //printf("i=%%d, j=%%d |",i,j);
                                    
                                    
                                    %(KERNEL_ARGUMENT_DECL)s
                                    
                                      //KERNEL CODE START
                                      
                                      %(KERNEL)s
                                      
                                      //KERNEL CODE END
                                    
                                    
                                }
                                _j = q_list[_j];
                            }
                            _i=q_list[_i];
                        }

                    }
                }
            }
            
            
            return;
        }        
        
        
        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """
        cell.cell_list.check()
        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        if n is not None:
            print "warning option depreciated"
            #_N = n
        else:
            _N = cell.cell_list.cell_list[cell.cell_list.cell_list.end]



        '''Pass access descriptor to dat'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1])

        args = [ctypes.c_int(_N),
                self._domain.cell_array.ctypes_data,
                cell.cell_list.cell_list.ctypes_data]

        args.append(self.loop_timer.get_python_parameters())


        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data)
            else:
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']


        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


###############################################################################
# Neighbour list looping using NIII
###############################################################################

class PairLoopNeighbourList(_Base):

    _neighbour_list_dict = {}

    def __init__(self, potential=None, dat_dict=None, kernel=None, shell_cutoff=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC
        self.rc = None
        # self.rn = None

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        if type(shell_cutoff) is not logic.Distance:
            shell_cutoff = logic.Distance(shell_cutoff)
        self._rn = shell_cutoff

        self.loop_timer = opt.LoopTimer()
        self.wrapper_timer = opt.SynchronizedTimer(runtime.TIMER)

        # Init code
        self._kernel_code = self._kernel.code
        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)

        self._group = None

        for pd in self._particle_dat_dict.items():
            if issubclass(type(pd[1][0]), data.PositionDat):
                self._group = pd[1][0].group
                break

        assert self._group is not None, "No cell to particle map found"


        new_decomp_flag = self._group.get_domain().cell_decompose(self._rn.value)

        if new_decomp_flag:
            self._group.get_cell_to_particle_map().create()

        self._key = (self._rn, self._group.get_domain(), self._group.get_position_dat())

        _nd = PairLoopNeighbourList._neighbour_list_dict
        if not self._key in _nd.keys() or new_decomp_flag:
            _nd[self._key] = cell.NeighbourListv2(self._group.get_cell_to_particle_map())


            _nd[self._key].setup(self._group.get_npart_local_func(),
                                 self._group.get_position_dat(),
                                 self._group.get_domain(),
                                 self._rn.value)

        self.neighbour_list = _nd[self._key]

        self._neighbourlist_count = 0
        self._invocations = 0

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''

        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        extern "C" void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const long* %(RESTRICT)s START_POINTS, const int* %(RESTRICT)s NLIST, %(ARGUMENTS)s);

        '''
        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir,
             'RESTRICT': self._cc.restrict_keyword}
        return code % d

    def _kernel_argument_declarations(self):
        s = build.Code()
        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []


            s += generation.generate_map(pair=True,
                                         symbol_external=dat[0] + '_ext',
                                         symbol_internal=dat[0],
                                         dat=dat[1],
                                         access_type=_mode)

        return s.string

    def _code_init(self):
        self._kernel_code = self._kernel.code
        self._code = '''
        #include <stdio.h>

        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const long* %(RESTRICT)s START_POINTS, const int* %(RESTRICT)s NLIST, %(ARGUMENTS)s) {


            %(LOOP_TIMER_PRE)s


            for(int _i = 0; _i < N_LOCAL; _i++){
                for(long _k = START_POINTS[_i]; _k < START_POINTS[_i+1]; _k++){
                    int _j = NLIST[_k];
                    int _cpp_halo_flag;
                    int _cp_halo_flag = 0;

                    // set halo flag, TODO move all halo flags to be an if condition on particle index?
                    //if (_i >= N_LOCAL) { _cp_halo_flag = 1; } else { _cp_halo_flag = 0; }
                    if (_j >= N_LOCAL) { _cpp_halo_flag = 1; } else { _cpp_halo_flag = 0; }

                     %(KERNEL_ARGUMENT_DECL)s

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }
            }

            %(LOOP_TIMER_POST)s

            return;
        }

        '''

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and potential engery.
        """

        cell2part = self._group.get_cell_to_particle_map()
        cell2part.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict


        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)


        '''Pass access descriptor to dat'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1])


        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data)
            else:
                args.append(dat_orig.ctypes_data)


        '''Rebuild neighbour list potentially'''
        self._invocations += 1
        if cell2part.version_id > self.neighbour_list.version_id:
            self.neighbour_list.update()
            #print "new list"
            #print self.neighbour_list.neighbour_starting_points.data[0:16]
            #print self.neighbour_list.list.data[0:10:]



            self._neighbourlist_count += 1



        '''Create arg list'''
        _N_TOTAL = ctypes.c_int(self.neighbour_list.n_total)
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_TOTAL,
                 _N_LOCAL,
                 _STARTS,
                 _LIST]


        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


###############################################################################
# Neighbour list that should vectorise based on PairLoopNeighbourList
###############################################################################


class VectorPairLoopNeighbourList(PairLoopNeighbourList):
    def _code_init(self):
        self._cc = build.ICC

        self._kernel_code = self._kernel.code
        self._code = '''
        #include <stdio.h>

        #define _BLOCK_SIZE 8


        void %(KERNEL_NAME)s_wrapper(const int N_TOTAL, const int N_LOCAL, const int* %(RESTRICT)s START_POINTS, const int* %(RESTRICT)s NLIST, %(ARGUMENTS)s) {

            START_POINTS = (const int*) __builtin_assume_aligned(START_POINTS, 16);
            NLIST = (const int*) __builtin_assume_aligned(NLIST, 16);
            A_ext = (double *) __builtin_assume_aligned(A_ext, 16);
            P_ext = (const double *) __builtin_assume_aligned(P_ext, 16);
            u_ext = (double *) __builtin_assume_aligned(u_ext, 16);



            //these essentially become the reduction code to be generated.
            double Ai_tmp[_BLOCK_SIZE*3] __attribute__((aligned(16)));
            double Aj_tmp[_BLOCK_SIZE*3] __attribute__((aligned(16)));
            double u_tmp[_BLOCK_SIZE] __attribute__((aligned(16)));



            %(LOOP_TIMER_PRE)s


            #define P(x,y) P_##x(y)
            #define P_0(y) P_ext[_i*3 + (y)]
            #define P_1(y) P_ext[_j*3 + (y)]

            #define A(x,y) A_##x(y)
            #define A_0(y) Ai_tmp[_pxi + _BLOCK_SIZE * (y)]
            #define A_1(y) Aj_tmp[_pxi + _BLOCK_SIZE * (y)]

            #define u(x) u_##x
            #define u_0 u_tmp[_pxi]




            for(int _i = 0; _i < N_LOCAL; _i++){

                //printf("%%d \\n", _i);

                const int _NBLOCKS = (START_POINTS[_i+1] - START_POINTS[_i])/_BLOCK_SIZE;
                const int S0 = START_POINTS[_i];

                //zero the _i accel stores
                for(int _ti = 0; _ti < _BLOCK_SIZE*3; _ti++){ Ai_tmp[_ti] = 0.0; }



                //loop over the blocks
                for(int _bx = 0; _bx < _NBLOCKS; _bx++){


                    // zero the stores for this block
                    for(int _ti = 0; _ti < _BLOCK_SIZE*3; _ti++){ Aj_tmp[_ti] = 0.0; }
                    for(int _ti = 0; _ti < _BLOCK_SIZE; _ti++){ u_tmp[_ti] = 0.0; }



                    // apply kernel to this block (This should vectorise nicely)
                    #pragma simd
                    for(int _pxi = 0; _pxi < _BLOCK_SIZE; _pxi++){
                         const int _j = NLIST[_bx*_BLOCK_SIZE + _pxi + S0];

                         //printf("%%d \\n", _j);


                         //KERNEL CODE START

                         %(KERNEL)s

                         //KERNEL CODE END

                    }


                    //cleanup potential energy
                    double _u_red = 0.0, _u_red_h = 0.0;
                    for(int _ti=0; _ti < _BLOCK_SIZE; _ti++) {
                        ( ( NLIST[_bx*_BLOCK_SIZE + _ti + S0] < N_LOCAL ) ? _u_red : _u_red_h ) += u_tmp[_ti];
                    }
                    u_ext[0] += _u_red;
                    u_ext[1] += _u_red_h;


                    //cleanup _j accelerations
                    #pragma ivdep
                    for(int _pxi = 0; _pxi < _BLOCK_SIZE; _pxi++){
                        const int _j = NLIST[_bx*_BLOCK_SIZE + _pxi + S0];
                        if( _j < N_LOCAL ){

                            A_ext[_j*3]     += A(1, 0);
                            A_ext[_j*3 + 1] += A(1, 1);
                            A_ext[_j*3 + 2] += A(1, 2);

                        }
                    }




                    //end of all blocks for this particle
                }

                //reduce the _i accelerations


                double _A_red[3] __attribute__((aligned(16)));
                for(int _tx = 0; _tx < 3; _tx++){ _A_red[_tx] = 0.0; }

                #pragma ivdep
                for(int _tx = 0; _tx < 3; _tx++){
                    for(int _pxi = 0; _pxi < _BLOCK_SIZE; _pxi++){
                        _A_red[_tx] += A(0, _tx);
                    }
                }

                A_ext[_i*3]     += _A_red[0];
                A_ext[_i*3 + 1] += _A_red[1];
                A_ext[_i*3 + 2] += _A_red[2];




                #undef A_0
                #define A_0(y) A_ext[_i*3 + (y)]

                #undef A_1
                #define A_1(y) ( (_j < N_LOCAL) ? A_ext[_j*3 + (y)] : _null )

                #undef u_0
                #define u_0 ( (_j < N_LOCAL) ? u_ext[0] : u_ext[1] )

                for(int _k = START_POINTS[_i] + _NBLOCKS*_BLOCK_SIZE; _k < START_POINTS[_i+1]; _k++){
                    const int _j = NLIST[_k];

                     double _null;

                     //KERNEL CODE START

                     %(KERNEL)s

                     //KERNEL CODE END

                }

            }


            %(LOOP_TIMER_POST)s

            return;
        }

        '''

# #############################################################################
# --------------------------------- AST ---------------------------------------
# #############################################################################

def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)




class PairLoopNeighbourListNS(object):


    _neighbour_list_dict = {}

    def __init__(self, potential=None, dat_dict=None, kernel=None, shell_cutoff=None):

        self._potential = potential
        self._particle_dat_dict = dat_dict
        self._cc = build.TMPCC
        self.rc = None
        # self.rn = None

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        if potential is not None:
            self._kernel = self._potential.kernel
        elif kernel is not None:
            self._kernel = kernel
        else:
            print "pairloop error, no kernel passed."

        if type(shell_cutoff) is not logic.Distance:
            shell_cutoff = logic.Distance(shell_cutoff)
        self._rn = shell_cutoff

        self.loop_timer = opt.LoopTimer()
        self.wrapper_timer = opt.SynchronizedTimer(runtime.TIMER)


        self._components = {'LIB_PAIR_INDEX_0': '_i',
                            'LIB_PAIR_INDEX_1': '_j',
                            'LIB_NAME': str(self._kernel.name) + '_wrapper'}
        self._gather_size_limit = 4
        self._generate()


        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._components['LIB_SRC'],
                                             self._kernel.name,
                                             CC=self._cc)

        self._group = None

        for pd in self._particle_dat_dict.items():
            if issubclass(type(pd[1][0]), data.PositionDat):
                self._group = pd[1][0].group
                break

        assert self._group is not None, "No cell to particle map found"


        new_decomp_flag = self._group.get_domain().cell_decompose(self._rn.value)

        if new_decomp_flag:
            self._group.get_cell_to_particle_map().create()

        self._key = (self._rn, self._group.get_domain(), self._group.get_position_dat())

        _nd = PairLoopNeighbourList._neighbour_list_dict
        if not self._key in _nd.keys() or new_decomp_flag:
            _nd[self._key] = cell.NeighbourListNonN3(self._group.get_cell_to_particle_map())


            _nd[self._key].setup(self._group.get_npart_local_func(),
                                 self._group.get_position_dat(),
                                 self._group.get_domain(),
                                 self._rn.value)

        self.neighbour_list = _nd[self._key]

        self._neighbourlist_count = 0
        self._invocations = 0




    def _generate(self):
        self._generate_lib_specific_args()
        self._generate_kernel_arg_decls()
        self._generate_kernel_func()
        self._generate_map_macros()
        self._generate_kernel_headers()

        self._generate_kernel_gather()
        self._generate_kernel_call()
        self._generate_kernel_scatter()

        self._generate_lib_inner_loop_block()
        self._generate_lib_inner_loop()

        self._generate_lib_outer_loop()
        self._generate_lib_func()
        self._generate_lib_src()

        # print 60*"-"
        # print self._components['LIB_SRC']
        # print 60*"-"




    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, 'N_LOCAL')),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int64_str,
                               Restrict(self._cc.restrict_keyword,'START_POINTS'))
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, 'NLIST')),
                )
            ),
            self.loop_timer.get_cpp_arguments_ast()
        ]




    def _generate_kernel_arg_decls(self):

        _kernel_arg_decls = []
        _kernel_lib_arg_decls = []
        _kernel_structs = cgen.Module([
            cgen.Comment('#### Structs generated per ParticleDat ####')
        ])

        if self._kernel.static_args is not None:

            for i, dat in enumerate(self._kernel.static_args.items()):
                _kernel_arg_decls.append(
                    cgen.Const(cgen.Value(host.ctypes_map[dat[1]], dat[0]))
                )

        for i, dat in enumerate(self._particle_dat_dict.items()):

            assert type(dat[1]) is tuple, "Access descriptors not found"

            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dat[1][0].dtype],
                                          Restrict(self._cc.restrict_keyword, dat[0]))
                                      )

            if issubclass(type(dat[1][0]), host.Array):
                kernel_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dat[1][0].dtype],
                                              Restrict(self._cc.restrict_keyword, dat[0]))
                                          )
                if not dat[1][1].write:
                    kernel_arg = cgen.Const(kernel_arg)

                _kernel_arg_decls.append(kernel_arg)

            elif issubclass(type(dat[1][0]), host.Matrix):
                # MAKE STRUCT TYPE
                dtype = dat[1][0].dtype
                ti = cgen.Pointer(cgen.Value(cgen.dtype_to_ctype(dtype),
                                             Restrict(self._cc.restrict_keyword,'i')))
                tj = cgen.Pointer(cgen.Value(cgen.dtype_to_ctype(dtype),
                                             Restrict(self._cc.restrict_keyword,'j')))
                if not dat[1][1].write:
                    ti = cgen.Const(ti)
                    tj = cgen.Const(tj)
                typename = dat[0]+'_t'
                _kernel_structs.append(cgen.Typedef(cgen.Struct('', [ti,tj], typename)))


                # MAKE STRUCT ARG
                _kernel_arg_decls.append(cgen.Value(typename, dat[0]))


            if not dat[1][1].write:
                kernel_lib_arg = cgen.Const(kernel_lib_arg)

            _kernel_lib_arg_decls.append(kernel_lib_arg)

        self._components['KERNEL_ARG_DECLS'] = _kernel_arg_decls
        self._components['KERNEL_LIB_ARG_DECLS'] = _kernel_lib_arg_decls
        self._components['KERNEL_STRUCT_TYPEDEFS'] = _kernel_structs


    def _generate_map_macros(self):

        g = cgen.Module([cgen.Comment('#### KERNEL_MAP_MACROS ####')])

        for i, dat in enumerate(self._particle_dat_dict.items()):
            if issubclass(type(dat[1][0]), host.Array):
                g.append(cgen.Define(dat[0]+'(x)', '('+dat[0]+'[(x)])'))
            if issubclass(type(dat[1][0]), host.Matrix):
                g.append(cgen.Define(dat[0]+'(x,y)', dat[0]+'_##x(y)'))
                g.append(cgen.Define(dat[0]+'_0(y)', dat[0]+'.i[(y)]'))
                g.append(cgen.Define(dat[0]+'_1(y)', dat[0]+'.j[(y)]'))


        self._components['KERNEL_MAP_MACROS'] = g



    def _generate_kernel_func(self):
        self._components['KERNEL_FUNC'] = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.DeclSpecifier(
                    cgen.Value("void", 'k_' + self._kernel.name), 'inline'
                ),
                self._components['KERNEL_ARG_DECLS']
            ),
                cgen.Block([
                    cgen.Line(self._kernel.code)
                ])
            )



    def _generate_kernel_headers(self):
        s = []
        if self._kernel.headers is not None:
            for x in self._kernel.headers:
                s.append(cgen.Include(x, system=False))

        s.append(self.loop_timer.get_cpp_headers_ast())
        self._components['KERNEL_HEADERS'] = cgen.Module(s)





    def _generate_lib_inner_loop_block(self):
        self._components['LIB_INNER_LOOP_BLOCK'] = \
            cgen.Block([cgen.Line('const int ' + self._components['LIB_PAIR_INDEX_1']
                                  + ' = NLIST[_k];\n \n'),
                        self._components['LIB_KERNEL_CALL']
                        ])



    def _generate_lib_inner_loop(self):
        i = self._components['LIB_PAIR_INDEX_0']
        b = self._components['LIB_INNER_LOOP_BLOCK']
        self._components['LIB_INNER_LOOP'] = cgen.For('long _k=START_POINTS['+i+']',
                                                      '_k<START_POINTS['+i+'+1]',
                                                      '_k++',
                                                      b
                                                      )





    def _generate_kernel_gather(self):

        kernel_gather = cgen.Module([cgen.Comment('#### Pre kernel gather ####')])


        if self._kernel.static_args is not None:

            for i, dat in enumerate(self._kernel.static_args.items()):
                pass


        for i, dat in enumerate(self._particle_dat_dict.items()):

            if issubclass(type(dat[1][0]), host.Array):
                pass
            elif issubclass(type(dat[1][0]), host.Matrix) \
                    and dat[1][1].write \
                    and dat[1][0].ncomp <= self._gather_size_limit:


                isym = dat[0]+'i'
                nc = dat[1][0].ncomp
                ncb = '['+str(nc)+']'
                dtype = host.ctypes_map[dat[1][0].dtype]

                t = '{'
                for tx in range(nc):
                    t+= '*(' + dat[0] + '+' + self._components['LIB_PAIR_INDEX_0']
                    t+= '*' + str(nc) + '+' + str(tx) + '),'
                t = t[:-1] + '}'

                g = cgen.Value(dtype,isym+ncb)
                '''
                if not dat[1][1].write:
                    g = cgen.Const(g)
                '''
                g = cgen.Initializer(g,t)



                kernel_gather.append(g)



        self._components['LIB_KERNEL_GATHER'] = kernel_gather




    def _generate_kernel_call(self):

        kernel_call = cgen.Module([cgen.Comment('#### Kernel call arguments ####')])
        kernel_call_symbols = []

        for i, dat in enumerate(self._particle_dat_dict.items()):
            if issubclass(type(dat[1][0]), host.Array):
                kernel_call_symbols.append(dat[0])
            elif issubclass(type(dat[1][0]), host.Matrix):
                call_symbol = dat[0] + '_c'
                kernel_call_symbols.append(call_symbol)

                nc = str(dat[1][0].ncomp)
                _ishift = '+' + self._components['LIB_PAIR_INDEX_0'] + '*' + nc
                _jshift = '+' + self._components['LIB_PAIR_INDEX_1'] + '*' + nc

                if dat[1][1].write and dat[1][0].ncomp <= self._gather_size_limit:
                    isym = '&'+ dat[0]+'i[0]'
                else:
                    isym = dat[0] + _ishift
                jsym = dat[0] + _jshift
                g = cgen.Value(dat[0]+'_t', call_symbol)
                g = cgen.Initializer(g, '{ ' + isym + ', ' + jsym + '}')

                kernel_call.append(g)



            else:
                print "ERROR: Type not known"

        kernel_call.append(cgen.Comment('#### Kernel call ####'))

        kernel_call_symbols_s = ''
        for sx in kernel_call_symbols:
            kernel_call_symbols_s += sx +','
        kernel_call_symbols_s=kernel_call_symbols_s[:-1]

        kernel_call.append(cgen.Line(
            'k_'+self._kernel.name+'(' + kernel_call_symbols_s + ');'
        ))

        self._components['LIB_KERNEL_CALL'] = kernel_call




    def _generate_kernel_scatter(self):

        kernel_scatter = cgen.Module([cgen.Comment('#### Post kernel scatter ####')])


        if self._kernel.static_args is not None:

            for i, dat in enumerate(self._kernel.static_args.items()):
                pass


        for i, dat in enumerate(self._particle_dat_dict.items()):

            if issubclass(type(dat[1][0]), host.Array):
                pass
            elif issubclass(type(dat[1][0]), host.Matrix)\
                    and dat[1][1].write\
                    and dat[1][0].ncomp <= self._gather_size_limit:

                
                isym = dat[0]+'i'
                nc = dat[1][0].ncomp
                ncb = '['+str(nc)+']'
                dtype = host.ctypes_map[dat[1][0].dtype]
                ix =self._components['LIB_PAIR_INDEX_0']

                b = cgen.Assign(dat[0]+'['+str(nc)+'*'+ix+'+tx]', isym+'[tx]')
                g = cgen.For('int tx=0', 'tx<'+str(nc), 'tx++',
                             cgen.Block([b]))


                kernel_scatter.append(g)



        self._components['LIB_KERNEL_SCATTER'] = kernel_scatter





    def _generate_lib_outer_loop(self):

        block = cgen.Block([self._components['LIB_KERNEL_GATHER'],
                            self._components['LIB_INNER_LOOP'],
                            self._components['LIB_KERNEL_SCATTER']])

        i = self._components['LIB_PAIR_INDEX_0']

        loop = cgen.For('int ' + i + '=0',
                        i + '<N_LOCAL',
                        i+'++',
                        block)

        self._components['LIB_OUTER_LOOP'] = loop


    def _generate_lib_func(self):
        block = cgen.Block([
            self.loop_timer.get_cpp_pre_loop_code_ast(),
            self._components['LIB_OUTER_LOOP'],
            self.loop_timer.get_cpp_post_loop_code_ast()
        ])


        self._components['LIB_FUNC'] = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", self._components['LIB_NAME'])
            ,
                self._components['LIB_ARG_DECLS'] + self._components['KERNEL_LIB_ARG_DECLS']
            ),
                block
            )

    def _generate_lib_src(self):
        self._components['LIB_SRC'] = cgen.Module([
            self._components['KERNEL_STRUCT_TYPEDEFS'],
            self._components['KERNEL_MAP_MACROS'],
            cgen.Comment('#### Kernel function ####'),
            self._components['KERNEL_FUNC'],
            cgen.Comment('#### Library function ####'),
            self._components['LIB_FUNC']
        ])



    def _generate_header_source(self):
        """Generate the source code of the header file.
        Returns the source code for the header file.
        """
        code = '''
        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        extern "C" %(FUNC_DEC)s
        '''
        d = {'INCLUDED_HEADERS': str(self._components['KERNEL_HEADERS']),
             'FUNC_DEC': str(self._components['LIB_FUNC'].fdecl),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d



    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and
         potential engery.
        """

        cell2part = self._group.get_cell_to_particle_map()
        cell2part.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict


        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)


        '''Pass access descriptor to dat'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1])


        '''Add pointer arguments to launch command'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data)
            else:
                args.append(dat_orig.ctypes_data)


        '''Rebuild neighbour list potentially'''
        self._invocations += 1
        if cell2part.version_id > self.neighbour_list.version_id:
            self.neighbour_list.update()
            #print "new list"
            #print self.neighbour_list.neighbour_starting_points.data[0:16]
            #print self.neighbour_list.list.data[0:10:]



            self._neighbourlist_count += 1


        '''Create arg list'''
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_LOCAL,
                 _STARTS,
                 _LIST]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()

