# system level
import ppmd.modules.code_timer
import ppmd.opt
import ppmd.pairloop.neighbourlist_27cell
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import cgen
import os

from ppmd.pairloop.base import *
# package level
from ppmd import data, runtime, access
from ppmd.pairloop import neighbourlist_14cell
from ppmd.pairloop import neighbourlist_27cell
from ppmd.lib.common import ctypes_map


def gather_matrix(obj, symbol_dat, symbol_tmp, loop_index):
    nc = obj.ncomp
    t = '{'
    for tx in range(nc):
        t+= '*(' + symbol_dat + '+' + loop_index
        t+= '*' + str(nc) + '+' + str(tx) + '),'
    t = t[:-1] + '}'

    g = cgen.Value(obj.ctype, symbol_tmp + '['+str(nc)+']')
    return cgen.Initializer(g,t)

def scatter_matrix(obj, symbol_dat, symbol_tmp, loop_index):
    nc = obj.ncomp

    b = cgen.Assign(symbol_dat+'[' + str(nc) + '*' + loop_index + '+_tx]',
                    symbol_tmp + '[_tx]')
    g = cgen.For('int _tx=0', '_tx<' + str(nc), '_tx++',
                 cgen.Block([b]))
    return g


class PairLoopNeighbourListNS(object):

    _neighbour_list_dict_PNLNS = {}

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):

        self._dat_dict = access.DatArgStore(
            self._get_allowed_types(),
            dat_dict
        )

        self._cc = build.TMPCC

        self._temp_dir = runtime.BUILD_DIR
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        self._kernel = kernel

        self.shell_cutoff = shell_cutoff

        self.loop_timer = ppmd.modules.code_timer.LoopTimer()
        self.wrapper_timer = ppmd.opt.Timer(runtime.TIMER)
        self.list_timer = ppmd.opt.Timer(runtime.TIMER)

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
        self._make_neigbour_list()

        self._neighbourlist_count = 0
        self._kernel_execution_count = 0
        self._invocations = 0

    @staticmethod
    def _get_n_dict():
        return PairLoopNeighbourListNS._neighbour_list_dict_PNLNS


    def _neighbour_list_from_group(self, group):
        _nd = self._get_n_dict()
        # if flag is true then a new cell list was created
        flag = group.cell_decompose(self.shell_cutoff)

        self._key = (self.shell_cutoff,
                     group.domain,
                     group.get_position_dat())

        if not self._key in _nd.keys():
            _nd[self._key] = (
                group.get_cell_to_particle_map().instance_id,
                self._new_neighbour_list(self.shell_cutoff, group)
            )

    @staticmethod
    def _new_neighbour_list(shell_cutoff, group):
        nl = ppmd.pairloop.neighbourlist_27cell.NeighbourListNonN3(
            group.get_cell_to_particle_map()
        )
        nl.setup(group.get_npart_local_func(),
                 group.get_position_dat(),
                 group.domain,
                 shell_cutoff)
        return nl

    def _make_neigbour_list(self):
        if self._group is not None:
            self._neighbour_list_from_group(self._group)

    def _get_neighbour_list(self, group):
        key = (self.shell_cutoff,
               group.domain,
               group.get_position_dat())

        nd = self._get_n_dict()
        if nd[key][0] < group.get_cell_to_particle_map().instance_id:
            # need to remake neighbourlist
            nd[key] = (
                group.get_cell_to_particle_map().instance_id,
                self._new_neighbour_list(self.shell_cutoff, group)
            )

        return nd[key][1]

    @staticmethod
    def _get_allowed_types():
        return {
            data.ScalarArray: access.all_access_types,
            data.ParticleDat: access.all_access_types,
            data.PositionDat: access.all_access_types,
            data.GlobalArrayClassic: (access.INC_ZERO, access.INC, access.READ),
            data.GlobalArrayShared: (access.INC_ZERO, access.INC, access.READ),
        }

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
                    cgen.Value(host.long_str,
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
                arg = cgen.Const(cgen.Value(host.ctypes_map[dat[1]], dat[0]))
                _kernel_arg_decls.append(arg)
                _kernel_lib_arg_decls.append(arg)

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
                ti = cgen.Pointer(cgen.Value(ctypes_map(dtype),
                                             Restrict(self._cc.restrict_keyword,'i')))
                tj = cgen.Pointer(cgen.Value(ctypes_map(dtype),
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
                g = gather_matrix(dat[1][0], dat[0], isym,
                        self._components['LIB_PAIR_INDEX_0'])
                kernel_gather.append(g)

        self._components['LIB_KERNEL_GATHER'] = kernel_gather

    def _generate_kernel_scatter(self):
        kernel_scatter = cgen.Module([cgen.Comment('#### Post kernel scatter ####')])

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host.Matrix)\
                    and dat[1][1].write\
                    and dat[1][0].ncomp <= self._gather_size_limit:

                isym = dat[0]+'i'
                ix =self._components['LIB_PAIR_INDEX_0']
                g = scatter_matrix(dat[1][0], dat[0], isym, ix)
                kernel_scatter.append(g)

        self._components['LIB_KERNEL_SCATTER'] = kernel_scatter


    def _generate_kernel_call(self):

        kernel_call = cgen.Module([cgen.Comment('#### Kernel call arguments ####')])
        kernel_call_symbols = []
        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                kernel_call_symbols.append(dat[0])

        for i, dat in enumerate(self._dat_dict.items()):
            ast = None
            if issubclass(type(dat[1][0]), host._Array):
                call_symbol = dat[0]
            elif issubclass(type(dat[1][0]), host.Matrix):
                call_symbol = dat[0] + '_c'

                nc = str(dat[1][0].ncomp)
                _ishift = '+' + self._components['LIB_PAIR_INDEX_0'] + '*' + nc
                if dat[1][1].write and dat[1][0].ncomp <= self._gather_size_limit:
                    isym = '&'+ dat[0]+'i[0]'
                else:
                    isym = dat[0] + _ishift

                _jshift = '+' + self._components['LIB_PAIR_INDEX_1'] + '*' + nc

                jsym = dat[0] + _jshift
                g = cgen.Value('_'+dat[0]+'_t', call_symbol)
                g = cgen.Initializer(g, '{ ' + isym + ', ' + jsym + '}')

                ast = g

            else:
                raise RuntimeError("ERROR: Type not known")

            kernel_call_symbols.append(call_symbol)
            if ast is not None:
                kernel_call.append(ast)



        kernel_call.append(cgen.Comment('#### Kernel call ####'))

        kernel_call_symbols_s = ''
        for sx in kernel_call_symbols:
            kernel_call_symbols_s += sx +','
        kernel_call_symbols_s=kernel_call_symbols_s[:-1]

        kernel_call.append(cgen.Line(
            'k_'+self._kernel.name+'(' + kernel_call_symbols_s + ');'
        ))

        self._components['LIB_KERNEL_CALL'] = kernel_call


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


    def _get_class_lib_args(self, neighbour_list):
        _N_LOCAL = ctypes.c_int(neighbour_list.n_local)
        _STARTS = neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = neighbour_list.list.ctypes_data
        return [_N_LOCAL, _STARTS, _LIST,
                self.loop_timer.get_python_parameters()]

    def _get_static_lib_args(self, static_args):
        args = []

        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            for dat in self._kernel.static_args.get_args(static_args):
                args.append(dat)

        return args


    def _init_dat_lib_args(self, dats):
        """
        The halo exchange process may reallocate all in the group. Hence this 
        function loops over all dats to ensure pointers collected in the second
        pass over the dats are valid.
        """
        for dat_orig in self._dat_dict.values(dats):
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1], pair=True)


    def _get_dat_lib_args(self, dats):
        args = []
        for dat_orig in self._dat_dict.values(dats):
            if type(dat_orig) is tuple:

                obj = dat_orig[0]
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1], pair=True))
            else:
                raise RuntimeError
        return args

    def _post_execute_dats(self, dats):
        # post execution data access
        for dat_orig in self._dat_dict.values(dats):
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()


    def execute(self, n=None, dat_dict=None, static_args=None):

        _group = self._group # could be none
        if self._group is None:
            for pd in self._dat_dict.items(dat_dict):
                if issubclass(type(pd[1][0]), data.PositionDat):
                    _group = pd[1][0].group
                    break
            assert _group is not None, "no group found"
            self._neighbour_list_from_group(_group)

        neighbour_list = self._get_neighbour_list(_group)

        if neighbour_list.check_lib_rebuild():
            del self._get_n_dict()[_key]
            self._neighbour_list_from_group(_group)
            neighbour_list = self._get_neighbour_list(_group)

        cell2part = _group.get_cell_to_particle_map()
        cell2part.check()

        self._init_dat_lib_args(dat_dict)
        args = self._get_dat_lib_args(dat_dict)

        '''Rebuild neighbour list potentially'''
        self._invocations += 1

        self.list_timer.start()
        neighbour_list.update_if_required()
        self.list_timer.pause()

        '''Create arg list'''
        args = self._get_class_lib_args(neighbour_list) + \
               self._get_static_lib_args(static_args) + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        self._kernel_execution_count += neighbour_list.neighbour_starting_points[neighbour_list.n_local]

        self._update_opt()
        self._post_execute_dats(dat_dict)


###############################################################################
# Neighbour list looping using NIII
###############################################################################

class PairLoopNeighbourList(PairLoopNeighbourListNS):

    _neighbour_list_dict_PNL = {}

    @staticmethod
    def _get_n_dict():
        return PairLoopNeighbourList._neighbour_list_dict_PNL

    @staticmethod
    def _new_neighbour_list(shell_cutoff, group):
        nl = neighbourlist_14cell.NeighbourListv2(
                group.get_cell_to_particle_map()
        )
        nl.setup(group.get_npart_local_func(),
                 group.get_position_dat(),
                 group.domain,
                 shell_cutoff
        )
        return nl

    def execute(self, n=None, dat_dict=None, static_args=None):

        neighbour_list = self._get_neighbour_list(self._group)

        cell2part = self._group.get_cell_to_particle_map()
        cell2part.check()

        args = []
        # Add static arguments to launch command
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            args += self._kernel.static_args.get_args(static_args)

        # Add pointer arguments to launch command
        for dat in self._dat_dict.values(new_dats=dat_dict):
            obj = dat[0]
            mode = dat[1]
            obj.ctypes_data_access(mode, pair=True)

        # Add pointer arguments to launch command
        for dat in self._dat_dict.values(new_dats=dat_dict):
            obj = dat[0]
            mode = dat[1]
            args.append(obj.ctypes_data_access(mode, pair=True))

        # Rebuild neighbour list potentially
        self._invocations += 1
        if cell2part.version_id > neighbour_list.version_id:
            neighbour_list.update()
            self._neighbourlist_count += 1


        # Create arg list for lib
        _N_LOCAL = ctypes.c_int(neighbour_list.n_local)
        _STARTS = neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = neighbour_list.list.ctypes_data

        args2 = [_N_LOCAL, _STARTS, _LIST]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        # Execute the kernel over all particle pairs.
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        self._kernel_execution_count += \
            neighbour_list.neighbour_starting_points[neighbour_list.n_local]

        self._update_opt()
        self._post_execute_dats(dat_dict)

