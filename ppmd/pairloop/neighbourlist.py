# system level

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import os
import cgen

# package level

from ppmd import data, runtime, cell

from base import *


class PairLoopNeighbourListNS(object):


    _neighbour_list_dict_PNLNS = {}

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):

        self._dat_dict = dat_dict
        self._cc = build.TMPCC


        self._temp_dir = runtime.BUILD_DIR
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        self._kernel = kernel

        self.shell_cutoff = shell_cutoff


        self.loop_timer = opt.LoopTimer()
        self.wrapper_timer = opt.Timer(runtime.TIMER)
        self.list_timer = opt.Timer(runtime.TIMER)

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

        #assert self._group is not None, "No cell to particle map found"

        if self._group is not None:
            _nd = PairLoopNeighbourListNS._neighbour_list_dict_PNLNS
            # if flag is true then a new cell list was created
            flag = self._group.cell_decompose(self.shell_cutoff)

            if flag:
                for key in _nd.keys():
                    _nd[key] = cell.NeighbourListNonN3(
                        self._group.get_cell_to_particle_map()
                    )
                    _nd[key].setup(self._group.get_npart_local_func(),
                                   self._group.get_position_dat(),
                                   self._group.domain,
                                   key[0])

            self._key = (self.shell_cutoff,
                         self._group.domain,
                         self._group.get_position_dat())


            if not self._key in _nd.keys():

                _nd[self._key] = cell.NeighbourListNonN3(
                    self._group.get_cell_to_particle_map()
                )

                _nd[self._key].setup(self._group.get_npart_local_func(),
                                     self._group.get_position_dat(),
                                     self._group.domain,
                                     self.shell_cutoff)


        self._neighbourlist_count = 0
        self._kernel_execution_count = 0
        self._invocations = 0


    def _init_components(self):
         self._components = {
             'LIB_PAIR_INDEX_0': '_i',
             'LIB_PAIR_INDEX_1': '_j',
             'LIB_NAME': str(self._kernel.name) + '_wrapper',
             'LIB_HEADERS': [],
         }


    def _generate(self):
        self._init_components()
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
        s = self._components['LIB_HEADERS']
        if self._kernel.headers is not None:
            if hasattr(self._kernel.headers, "__iter__"):
                for x in self._kernel.headers:
                    s.append(x.ast)
            else:
                s.append(self._kernel.headers.ast)

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

        for i, dat in enumerate(self._dat_dict.items()):

            if issubclass(type(dat[1][0]), host.Matrix) \
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
                print("ERROR: Type not known")

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

    def _update_opt(self):
        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':list_timer'
        ] = self.list_timer.time()

        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':execute_internal'
        ] = self.loop_timer.time

        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':kernel_execution_count'
        ] =  self._kernel_execution_count


    def _get_class_lib_args(self):
        neighbour_list = PairLoopNeighbourListNS._neighbour_list_dict_PNLNS[self._key]
        _N_LOCAL = ctypes.c_int(neighbour_list.n_local)
        _STARTS = neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = neighbour_list.list.ctypes_data

        return [_N_LOCAL, _STARTS, _LIST, self.loop_timer.get_python_parameters()]

    def _get_static_lib_args(self, static_args):
        args = []

        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            for dat in static_args.values():
                args.append(dat)

        return args


    def __init_dat_lib_args(self, dats):
        for dat_orig in dats.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1], pair=True)


    def _get_dat_lib_args(self, dats):
        args = []
        for dat_orig in dats.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1], pair=True))
            else:
                raise RuntimeError
        return args

    def _post_execute_dats(self, dats):
        '''afterwards access descriptors'''
        for dat_orig in dats.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and
         potential engery.
        """

        '''Allow alternative pointers'''
        if dat_dict is not None:
            self._dat_dict = dat_dict

        _group = self._group
        if self._group is None:

            _group = None
            for pd in self._dat_dict.items():
                if issubclass(type(pd[1][0]), data.PositionDat):
                    _group = pd[1][0].group
                    break

            assert _group is not None, "no group found"

            _nd = PairLoopNeighbourListNS._neighbour_list_dict_PNLNS
            # if flag is true then a new cell list was created
            flag = _group.cell_decompose(self.shell_cutoff)

            if flag:
                for key in _nd.keys():
                    _nd[key] = cell.NeighbourListNonN3(
                        _group.get_cell_to_particle_map()
                    )
                    _nd[key].setup(_group.get_npart_local_func(),
                                   _group.get_position_dat(),
                                   _group.domain,
                                   key[0])


        self._key = (self.shell_cutoff,
                     _group.domain,
                     _group.get_position_dat())


        if self._group is None:
            if not self._key in _nd.keys():

                _nd[self._key] = cell.NeighbourListNonN3(
                    _group.get_cell_to_particle_map()
                )

                _nd[self._key].setup(_group.get_npart_local_func(),
                                     _group.get_position_dat(),
                                     _group.domain,
                                     self.shell_cutoff)

        neighbour_list = PairLoopNeighbourListNS._neighbour_list_dict_PNLNS[self._key]

        cell2part = _group.get_cell_to_particle_map()
        cell2part.check()

        self.__init_dat_lib_args(self._dat_dict)
        args = self._get_dat_lib_args(self._dat_dict)


        '''Rebuild neighbour list potentially'''
        self._invocations += 1

        self.list_timer.start()
        if cell2part.version_id > neighbour_list.version_id:
            neighbour_list.update()
            self._neighbourlist_count += 1
        self.list_timer.pause()

        '''Create arg list'''
        args = self._get_class_lib_args() + self._get_static_lib_args(static_args) + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        self._kernel_execution_count += neighbour_list.neighbour_starting_points[neighbour_list.n_local]

        self._update_opt()
        self._post_execute_dats(self._dat_dict)


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


        # if group is none there is no cell to particle map.
        # therefore no halo exchange etc
        assert self._group is not None, "No cell to particle map found"

        _nd = PairLoopNeighbourList._neighbour_list_dict_PNL

        # if flag is true then a new cell list was created
        flag = self._group.cell_decompose(self.shell_cutoff)

        if flag:
            for key in _nd.keys():
                _nd[key] = cell.NeighbourListv2(
                    self._group.get_cell_to_particle_map()
                )
                _nd[key].setup(self._group.get_npart_local_func(),
                               self._group.get_position_dat(),
                               self._group.domain,
                               key[0])

        self._key = (self.shell_cutoff,
                     self._group.domain,
                     self._group.get_position_dat())


        if not self._key in _nd.keys():

            _nd[self._key] = cell.NeighbourListv2(
                self._group.get_cell_to_particle_map()
            )

            _nd[self._key].setup(self._group.get_npart_local_func(),
                                 self._group.get_position_dat(),
                                 self._group.domain,
                                 self.shell_cutoff)

        self._neighbourlist_count = 0
        self._invocations = 0
        self._kernel_execution_count = 0

    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and
         potential engery.
        """

        neighbour_list = PairLoopNeighbourList._neighbour_list_dict_PNL[self._key]

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
                dat_orig[0].ctypes_data_access(dat_orig[1], pair=True)


        '''Add pointer arguments to launch command'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1], pair=True))
            else:
                raise RuntimeError
                #args.append(dat_orig.ctypes_data)


        '''Rebuild neighbour list potentially'''
        self._invocations += 1



        if cell2part.version_id > neighbour_list.version_id:
            neighbour_list.update()
            self._neighbourlist_count += 1


        '''Create arg list'''
        _N_LOCAL = ctypes.c_int(neighbour_list.n_local)
        _STARTS = neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = neighbour_list.list.ctypes_data

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


        self._kernel_execution_count += neighbour_list.neighbour_starting_points[neighbour_list.n_local]


        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':kernel_execution_count'
        ] =  self._kernel_execution_count


        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()

