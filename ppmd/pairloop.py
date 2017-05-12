# system level

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import ctypes
import os
import cgen

# package level
import data
import build
import runtime
import access
import cell
import host
import opt

class _Base(object):
    def __init__(self, n, kernel, dat_dict):

        self._cc = build.TMPCC

        self._N = n

        self._kernel = kernel

        self._dat_dict = dat_dict

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


        for i, dat in enumerate(self._dat_dict.items()):



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
                s += '#include \"' + str(x) + '\" \n'

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
            self._dat_dict = dat_dict

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
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]))
            else:
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']
        method(*args)

        '''after wards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])



###############################################################################
# RAPAPORT LOOP SERIAL FOR HALO DOMAINS
###############################################################################
def get_type_map_symbol():
    return '_TYPE_MAP'

def get_first_index_symbol():
    return '_i'

def get_second_index_symbol():
    return '_j'

def get_first_cell_is_halo_symbol():
    return '_cp_halo_flag'

def get_second_cell_is_halo_symbol():
    return '_cpp_halo_flag'

_nl = '\n'

def _generate_map(pair=True, symbol_external=None, symbol_internal=None, dat=None, access_type=None):
    """
    Create pointer map.

    :param pair:
    :param symbol_external:
    :param symbol_internal:
    :param dat:
    :param access_type:
    :param n3_disable_dats:
    :return:
    """

    assert symbol_external is not None, "generate_map error: No symbol_external"
    assert symbol_internal is not None, "generate_map error: No symbol_internal"
    assert dat is not None, "generate_map error: No dat"
    assert access_type is not None, "generate_map error: No access_type"

    _space = ' ' * 14

    if pair:
        _n = 2
    else:
        _n = 1

    _c = build.Code()
    _c += '#undef ' + symbol_internal + _nl

    if type(dat) is data.TypedDat:
        #case for typed dat
        _ncomp = dat.ncol

        # Define the entry point into the map
        _c += '#define ' + symbol_internal + '(x,y) ' + symbol_internal + '_##x(y)' + _nl

        # define the first particle map
        _c += '#define ' + symbol_internal + '_0(y) ' + symbol_external + \
              '[LINIDX_2D(' + str(_ncomp) + ',' + get_first_index_symbol() + \
              ',' + '(y)]' + _nl
        # second particle map
        _c += '#define ' + symbol_internal + '_1(y) ' + symbol_external + \
              '[LINIDX_2D(' + str(_ncomp) + ',' + get_second_index_symbol() + \
              ',' + '(y)]' + _nl

        return _c

class PairLoopRapaportHalo(_Base):
    def __init__(self, domain, potential=None, dat_dict=None, kernel=None):
        self._domain = domain
        self._potential = potential
        self._dat_dict = dat_dict
        self._cc = build.TMPCC

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR
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
        for i, dat_orig in enumerate(self._dat_dict.items()):

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

            s += _generate_map(pair=True,
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
             'LIB_DIR': runtime.LIB_DIR}
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
            self._dat_dict = dat_dict

        if n is not None:
            print "warning option depreciated"
            #_N = n
        else:
            _N = cell.cell_list.cell_list[cell.cell_list.cell_list.end]



        '''Pass access descriptor to dat'''
        for dat_orig in self._dat_dict.values():
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
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data)
            else:
                args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']


        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


# #############################################################################
# --------------------------------- AST ---------------------------------------
# #############################################################################

def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)


class PairLoopNeighbourListNS(object):


    _neighbour_list_dict_PNLNS = {}

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):

        self._dat_dict = dat_dict
        self._cc = build.TMPCC
        # self.rn = None

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        self._kernel = kernel

        '''
        if type(shell_cutoff) is not logic.Distance:
            shell_cutoff = logic.Distance(shell_cutoff)
        self.shell_cutoff = shell_cutoff
        '''
        self.shell_cutoff = shell_cutoff


        self.loop_timer = opt.LoopTimer()
        self.wrapper_timer = opt.Timer(runtime.TIMER)


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

        for pd in self._dat_dict.items():
            if issubclass(type(pd[1][0]), data.PositionDat):
                self._group = pd[1][0].group
                break

        assert self._group is not None, "No cell to particle map found"


        new_decomp_flag = self._group.get_domain().cell_decompose(
            self.shell_cutoff
        )

        if new_decomp_flag:
            self._group.get_cell_to_particle_map().create()

        self._key = (self.shell_cutoff,
                     self._group.get_domain(),
                     self._group.get_position_dat())


        _nd = PairLoopNeighbourListNS._neighbour_list_dict_PNLNS
        if not self._key in _nd.keys() or new_decomp_flag:
            _nd[self._key] = cell.NeighbourListNonN3(
                self._group.get_cell_to_particle_map()
            )


            _nd[self._key].setup(self._group.get_npart_local_func(),
                                 self._group.get_position_dat(),
                                 self._group.get_domain(),
                                 self.shell_cutoff)

        self.neighbour_list = _nd[self._key]

        self._neighbourlist_count = 0
        self._kernel_execution_count = 0
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
            cgen.Const(cgen.Value(host.int32_str, '_N_LOCAL')),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int64_str,
                               Restrict(
                                   self._cc.restrict_keyword,'_START_POINTS'
                               )
                               )
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_NLIST')),
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

        for i, dat in enumerate(self._dat_dict.items()):

            assert type(dat[1]) is tuple, "Access descriptors not found"

            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dat[1][0].dtype],
                                          Restrict(self._cc.restrict_keyword, dat[0]))
                                      )

            if issubclass(type(dat[1][0]), host._Array):
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
                typename = '_'+dat[0]+'_t'
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

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
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
                s.append(x.ast)

        s.append(self.loop_timer.get_cpp_headers_ast())
        self._components['KERNEL_HEADERS'] = cgen.Module(s)



    def _generate_lib_inner_loop_block(self):
        self._components['LIB_INNER_LOOP_BLOCK'] = \
            cgen.Block([cgen.Line('const int ' + self._components['LIB_PAIR_INDEX_1']
                                  + ' = _NLIST[_k];\n \n'),
                        self._components['LIB_KERNEL_CALL']
                        ])



    def _generate_lib_inner_loop(self):
        i = self._components['LIB_PAIR_INDEX_0']
        b = self._components['LIB_INNER_LOOP_BLOCK']
        self._components['LIB_INNER_LOOP'] = cgen.For('long _k=_START_POINTS['+i+']',
                                                      '_k<_START_POINTS['+i+'+1]',
                                                      '_k++',
                                                      b
                                                      )



    def _generate_kernel_gather(self):

        kernel_gather = cgen.Module([cgen.Comment('#### Pre kernel gather ####')])


        if self._kernel.static_args is not None:

            for i, dat in enumerate(self._kernel.static_args.items()):
                pass


        for i, dat in enumerate(self._dat_dict.items()):

            if issubclass(type(dat[1][0]), host._Array):
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

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
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
                g = cgen.Value('_'+dat[0]+'_t', call_symbol)
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


        for i, dat in enumerate(self._dat_dict.items()):

            if issubclass(type(dat[1][0]), host._Array):
                pass
            elif issubclass(type(dat[1][0]), host.Matrix)\
                    and dat[1][1].write\
                    and dat[1][0].ncomp <= self._gather_size_limit:

                
                isym = dat[0]+'i'
                nc = dat[1][0].ncomp
                ncb = '['+str(nc)+']'
                dtype = host.ctypes_map[dat[1][0].dtype]
                ix =self._components['LIB_PAIR_INDEX_0']

                b = cgen.Assign(dat[0]+'['+str(nc)+'*'+ix+'+_tx]', isym+'[_tx]')
                g = cgen.For('int _tx=0', '_tx<'+str(nc), '_tx++',
                             cgen.Block([b]))


                kernel_scatter.append(g)



        self._components['LIB_KERNEL_SCATTER'] = kernel_scatter





    def _generate_lib_outer_loop(self):

        block = cgen.Block([self._components['LIB_KERNEL_GATHER'],
                            self._components['LIB_INNER_LOOP'],
                            self._components['LIB_KERNEL_SCATTER']])

        i = self._components['LIB_PAIR_INDEX_0']

        loop = cgen.For('int ' + i + '=0',
                        i + '<_N_LOCAL',
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
             'LIB_DIR': runtime.LIB_DIR}
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
            self._dat_dict = dat_dict


        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            for dat in static_args.values():
                args.append(dat)


        '''Pass access descriptor to dat'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1])


        '''Add pointer arguments to launch command'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data)
            else:
                args.append(dat_orig.ctypes_data)


        '''Rebuild neighbour list potentially'''
        self._invocations += 1

        if cell2part.version_id > self.neighbour_list.version_id:

            self.neighbour_list.update()

            self._neighbourlist_count += 1


        '''Create arg list'''
        _N_LOCAL = ctypes.c_int(self.neighbour_list.n_local)
        _STARTS = self.neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = self.neighbour_list.list.ctypes_data

        args2 = [_N_LOCAL, _STARTS, _LIST]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()


        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':execute_internal'
        ] = self.loop_timer.time
        


        self._kernel_execution_count += self.neighbour_list.neighbour_starting_points[self.neighbour_list.n_local]
        

        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':kernel_execution_count'
        ] =  self._kernel_execution_count



        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()





###############################################################################
# Neighbour list looping using NIII
###############################################################################

class PairLoopNeighbourList(PairLoopNeighbourListNS):

    _neighbour_list_dict_PNL = {}

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):

        self._dat_dict = dat_dict
        self._cc = build.TMPCC
        # self.rn = None

        ##########
        # End of Rapaport initialisations.
        ##########

        self._temp_dir = runtime.BUILD_DIR
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        self._kernel = kernel
        '''
        if type(shell_cutoff) is not logic.Distance:
            shell_cutoff = logic.Distance(shell_cutoff)
        '''
        self.shell_cutoff = shell_cutoff

        self.loop_timer = opt.LoopTimer()
        self.wrapper_timer = opt.Timer(runtime.TIMER)


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

        for pd in self._dat_dict.items():
            if issubclass(type(pd[1][0]), data.PositionDat):
                self._group = pd[1][0].group
                break

        assert self._group is not None, "No cell to particle map found"


        new_decomp_flag = self._group.get_domain().cell_decompose(
            self.shell_cutoff
        )

        if new_decomp_flag:
            self._group.get_cell_to_particle_map().create()

        self._key = (
            self.shell_cutoff, self._group.get_domain(),
            self._group.get_position_dat()
            )

        _nd = PairLoopNeighbourList._neighbour_list_dict_PNL
        if not self._key in _nd.keys() or new_decomp_flag:
            _nd[self._key] = cell.NeighbourListv2(
                self._group.get_cell_to_particle_map()
            )


            _nd[self._key].setup(self._group.get_npart_local_func(),
                                 self._group.get_position_dat(),
                                 self._group.get_domain(),
                                 self.shell_cutoff)

        self.neighbour_list = _nd[self._key]

        self._neighbourlist_count = 0
        self._invocations = 0
        self._kernel_execution_count = 0




###############################################################################
# All To All looping Non-Symmetric
###############################################################################



class AllToAllNS(object):


    def __init__(self, kernel=None, dat_dict=None):

        self._dat_dict = dat_dict
        self._cc = build.TMPCC


        self._temp_dir = runtime.BUILD_DIR
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        self._kernel = kernel


        self.loop_timer = opt.LoopTimer()
        self.wrapper_timer = opt.Timer(runtime.TIMER)


        self._components = {'LIB_PAIR_INDEX_0': '_i',
                            'LIB_PAIR_INDEX_1': '_j',
                            'LIB_NAME': str(self._kernel.name) + '_wrapper'}

        self._gather_size_limit = 8

        self._generate()


        self._group = None

        for pd in self._dat_dict.items():
            if issubclass(type(pd[1][0]), data.PositionDat):
                self._group = pd[1][0].group
                break

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._components['LIB_SRC'],
                                             self._kernel.name,
                                             CC=self._cc)



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

        #print 60*"-"
        #print self._components['LIB_SRC']
        #print 60*"-"




    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, '_N_LOCAL')),
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

        for i, dat in enumerate(self._dat_dict.items()):

            assert type(dat[1]) is tuple, "Access descriptors not found"

            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dat[1][0].dtype],
                                          Restrict(self._cc.restrict_keyword, dat[0]))
                                      )

            # print host.ctypes_map[dat[1][0].dtype], dat[1][0].dtype

            if issubclass(type(dat[1][0]), host._Array):
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
                typename = '_'+dat[0]+'_t'
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

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
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
                s.append(x.ast)

        s.append(self.loop_timer.get_cpp_headers_ast())
        self._components['KERNEL_HEADERS'] = cgen.Module(s)





    def _generate_lib_inner_loop_block(self):
        self._components['LIB_INNER_LOOP_BLOCK'] = \
            cgen.Block([
                        self._components['LIB_KERNEL_CALL']
                        ])



    def _generate_lib_inner_loop(self):
        i = self._components['LIB_PAIR_INDEX_0']
        j = self._components['LIB_PAIR_INDEX_1']
        b = self._components['LIB_INNER_LOOP_BLOCK']
        self._components['LIB_INNER_LOOP'] = cgen.Module([

            cgen.For(
                'int ' + j + '=0',
                j + '<' + i,
                j + '++',
                b
                ),

            cgen.For(
                'int ' + j + '=1+' + i,
                j + '< _N_LOCAL',
                j + '++',
                b
                ),
        ])


    def _generate_kernel_gather(self):

        kernel_gather = cgen.Module([cgen.Comment('#### Pre kernel gather ####')])


        if self._kernel.static_args is not None:

            for i, dat in enumerate(self._kernel.static_args.items()):
                pass


        for i, dat in enumerate(self._dat_dict.items()):

            if issubclass(type(dat[1][0]), host._Array):
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

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
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
                g = cgen.Value('_'+dat[0]+'_t', call_symbol)
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


        for i, dat in enumerate(self._dat_dict.items()):

            if issubclass(type(dat[1][0]), host._Array):
                pass
            elif issubclass(type(dat[1][0]), host.Matrix)\
                    and dat[1][1].write\
                    and dat[1][0].ncomp <= self._gather_size_limit:


                isym = dat[0]+'i'
                nc = dat[1][0].ncomp
                ncb = '['+str(nc)+']'
                dtype = host.ctypes_map[dat[1][0].dtype]
                ix =self._components['LIB_PAIR_INDEX_0']

                b = cgen.Assign(dat[0]+'['+str(nc)+'*'+ix+'+_tx]', isym+'[_tx]')
                g = cgen.For('int _tx=0', '_tx<'+str(nc), '_tx++',
                             cgen.Block([b]))


                kernel_scatter.append(g)



        self._components['LIB_KERNEL_SCATTER'] = kernel_scatter





    def _generate_lib_outer_loop(self):

        block = cgen.Block([self._components['LIB_KERNEL_GATHER'],
                            self._components['LIB_INNER_LOOP'],
                            self._components['LIB_KERNEL_SCATTER']])

        i = self._components['LIB_PAIR_INDEX_0']

        loop = cgen.For('int ' + i + '=0',
                        i + '<_N_LOCAL',
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
             'LIB_DIR': runtime.LIB_DIR}
        return code % d



    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and
         potential engery.
        """

        if self._group is not None:
            cell2part = self._group.get_cell_to_particle_map()
            cell2part.check()

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._dat_dict = dat_dict


        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Pass access descriptor to dat'''

        if self._group is not None:
            for dat_orig in self._dat_dict.values():
                if type(dat_orig) is tuple:
                    dat_orig[0].ctypes_data_access(dat_orig[1])


        '''Add pointer arguments to launch command'''
        _N = 0
        for dat_orig in self._dat_dict.values():
            args.append(dat_orig[0].ctypes_data)
            _N = dat_orig[0].npart_local


        '''Create arg list'''
        if n is not None:
            _N_LOCAL = n
        else:
            _N_LOCAL = _N

        args2 = [_N_LOCAL]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        '''afterwards access descriptors'''
        if self._group is not None:
            for dat_orig in self._dat_dict.values():
                dat_orig[0].ctypes_data_post(dat_orig[1])



###############################################################################
# All To All looping Symmetric
###############################################################################



class AllToAll(AllToAllNS):
    def _generate_lib_inner_loop(self):
        i = self._components['LIB_PAIR_INDEX_0']
        j = self._components['LIB_PAIR_INDEX_1']
        b = self._components['LIB_INNER_LOOP_BLOCK']
        self._components['LIB_INNER_LOOP'] = cgen.Module([
            cgen.For(
                'int ' + j + '=1+' + i,
                j + '< _N_LOCAL',
                j + '++',
                b
                )
        ])


















