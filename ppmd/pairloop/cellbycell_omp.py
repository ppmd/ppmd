# system level
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import cgen
import ctypes

# package level

from ppmd import data, runtime, host, access, modules, opt
from ppmd.lib import build
from ppmd.pairloop.neighbourlist import Restrict, scatter_matrix


_offsets = (
    (-1,1,-1),
    (-1,-1,-1),
    (-1,0,-1),
    (0,1,-1),
    (0,-1,-1),
    (0,0,-1),
    (1,0,-1),
    (1,1,-1),
    (1,-1,-1),

    (-1,1,0),
    (-1,0,0),
    (-1,-1,0),
    (0,-1,0),
    (0,0,0),
    (0,1,0),
    (1,0,0),
    (1,1,0),
    (1,-1,0),

    (-1,0,1),
    (-1,1,1),
    (-1,-1,1),
    (0,0,1),
    (0,1,1),
    (0,-1,1),
    (1,0,1),
    (1,1,1),
    (1,-1,1)
)


class CellByCellOMP(object):

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):

        self._dat_dict = access.DatArgStore(
            self._get_allowed_types(),
            dat_dict
        )

        self._cc = build.TMPCC
        self._kernel = kernel
        self.shell_cutoff = shell_cutoff

        self.loop_timer = modules.code_timer.LoopTimer()
        self.wrapper_timer = opt.Timer(runtime.TIMER)
        self.list_timer = opt.Timer(runtime.TIMER)

        self._gather_size_limit = 4
        self._generate()

        self._offset_list = host.Array(ncomp=27, dtype=ctypes.c_int)

        #print(self._components['LIB_SRC'])

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
            self._make_cell_list(self._group)

        self._kernel_execution_count = 0
        self._invocations = 0

        self._jstore = [host.Array(ncomp=100, dtype=ctypes.c_int) for tx in \
                        range(runtime.NUM_THREADS)]


    def _make_cell_list(self, group):
        # if flag is true then a new cell list was created
        flag = group.cell_decompose(self.shell_cutoff)
        self._make_offset_list(group)

    def _make_offset_list(self, group):
        ca = group.domain.cell_array
        for ofi, ofs in enumerate(_offsets):
            self._offset_list[ofi] = ofs[0] + ca[0]*ofs[1] + ca[0]*ca[1]*ofs[2]

    @staticmethod
    def _get_allowed_types():
        return {
            data.ScalarArray: (access.READ,),
            data.ParticleDat: access.all_access_types,
            data.PositionDat: access.all_access_types,
            data.GlobalArrayClassic: (access.INC_ZERO, access.INC, access.READ),
            data.GlobalArrayShared: (access.INC_ZERO, access.INC, access.READ),
        }

    def _generate(self):
        self._init_components()
        self._generate_lib_specific_args()
        self._generate_kernel_arg_decls()
        self._generate_kernel_func()
        self._generate_kernel_headers()

        self._generate_kernel_gather()
        self._generate_kernel_call()
        self._generate_kernel_scatter()

        self._generate_lib_inner_loop_block()
        self._generate_lib_inner_loop()

        self._generate_lib_outer_loop()
        self._generate_lib_func()
        self._generate_lib_src()

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
        i = self._components['LIB_PAIR_INDEX_0']
        j = self._components['LIB_PAIR_INDEX_1']
        self._components['LIB_INNER_LOOP_BLOCK'] = \
            cgen.Block([
                cgen.Line('const int _jcell = _icell + _OFFSET[_k];'),
                cgen.Line('int '+j+' = _CELL_LIST[_jcell + _LIST_OFFSET];' ),
                cgen.For(
                    'int _k2=0','_k2<_CCC[_jcell]','_k2++',
                    cgen.Block([
                        cgen.Line('if(%(I)s!=%(J)s){_JJSTORE[_nn++]=%(J)s;}'%\
                             {'I':i, 'J':j}),
                        cgen.Line(j+' = _CELL_LIST['+j+'];'),
                    ])
                ),
            ])

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

    def _init_components(self):
         self._components = {
             'LIB_PAIR_INDEX_0': '_i',
             'LIB_PAIR_INDEX_1': '_j',
             'LIB_NAME': str(self._kernel.name) + '_wrapper',
             'LIB_HEADERS': [cgen.Include('omp.h', system=True),],
             'OMP_THREAD_INDEX_SYM': '_threadid',
             'OMP_SHARED_SYMS': ['_CELL_LIST', '_OFFSET', '_CRL', '_CCC',
                                 '_JSTORE']
         }

    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, '_NUM_THREADS')),
            cgen.Const(cgen.Value(host.int32_str, '_N_LOCAL')),
            cgen.Const(cgen.Value(host.int32_str, '_LIST_OFFSET')),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(
                                   self._cc.restrict_keyword,'_CELL_LIST'
                               )
                    )
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_CRL')),
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_CCC')),
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_OFFSET')),
                )
            ),
            cgen.Pointer(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_JSTORE')),
                )
            ),
            self.loop_timer.get_cpp_arguments_ast()
        ]
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
                self._components['LIB_ARG_DECLS'] + \
                    self._components['KERNEL_LIB_ARG_DECLS']
            ),
                block
            )

    def _generate_lib_src(self):
        self._components['LIB_SRC'] = cgen.Module([
            self._components['KERNEL_STRUCT_TYPEDEFS'],
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
            self.__class__.__name__+':'+self._kernel.name+\
                ':kernel_execution_count'
        ] =  self._kernel_execution_count

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
            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            kernel_lib_arg = cgen.Pointer(
                cgen.Value(
                    host.ctypes_map[obj.dtype],
                    Restrict(self._cc.restrict_keyword, symbol)
            ))

            if issubclass(type(obj), data.GlobalArrayClassic):
                kernel_lib_arg = cgen.Pointer(kernel_lib_arg)

            if issubclass(type(obj), host._Array):
                kernel_arg = cgen.Pointer(
                    cgen.Value(
                        host.ctypes_map[obj.dtype],
                        Restrict(self._cc.restrict_keyword, symbol)
                ))

                if not mode.write:
                    kernel_arg = cgen.Const(kernel_arg)
                _kernel_arg_decls.append(kernel_arg)

                if mode.write is True:
                    assert issubclass(type(obj), data.GlobalArrayClassic), \
                        "global array must be a thread safe type for \
                        write access. Type is:" + str(type(obj))


            elif issubclass(type(dat[1][0]), host.Matrix):
                # MAKE STRUCT TYPE
                dtype = dat[1][0].dtype
                ti = cgen.Pointer(cgen.Value(
                    cgen.dtype_to_ctype(dtype),
                    Restrict(self._cc.restrict_keyword,'i')
                ))
                tj = cgen.Pointer(cgen.Value(
                    cgen.dtype_to_ctype(dtype),
                    Restrict(self._cc.restrict_keyword,'j')
                ))
                if not dat[1][1].write:
                    ti = cgen.Const(ti)
                    tj = cgen.Const(tj)
                typename = '_'+dat[0]+'_t'
                _kernel_structs.append(cgen.Typedef(cgen.Struct(
                    '', [ti,tj], typename)))

                # MAKE STRUCT ARG
                _kernel_arg_decls.append(cgen.Value(typename, dat[0]))

            if not dat[1][1].write:
                kernel_lib_arg = cgen.Const(kernel_lib_arg)

            _kernel_lib_arg_decls.append(kernel_lib_arg)

        self._components['KERNEL_ARG_DECLS'] = _kernel_arg_decls
        self._components['KERNEL_LIB_ARG_DECLS'] = _kernel_lib_arg_decls
        self._components['KERNEL_STRUCT_TYPEDEFS'] = _kernel_structs

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

    def _generate_kernel_gather(self):

        kernel_gather = cgen.Module([
            cgen.Comment('#### Pre kernel gather ####'),
            cgen.Initializer(cgen.Const(cgen.Value(
                'int', self._components['OMP_THREAD_INDEX_SYM'])),
                'omp_get_thread_num()')
        ])
        shared_syms = self._components['OMP_SHARED_SYMS']

        for i, dat in enumerate(self._dat_dict.items()):

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]
            shared_syms.append(symbol)

            if issubclass(type(obj), data.GlobalArrayClassic):
                isym = symbol+'_c'
                val = symbol+'['+ self._components['OMP_THREAD_INDEX_SYM'] +']'

                g = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype], isym))
                if not mode.write:
                    g = cgen.Const(g)
                g = cgen.Initializer(g, val)

                kernel_gather.append(g)

            elif issubclass(type(obj), host.Matrix) \
                    and mode.write \
                    and obj.ncomp <= self._gather_size_limit:


                isym = symbol+'i'
                nc = obj.ncomp
                ncb = '['+str(nc)+']'
                dtype = host.ctypes_map[obj.dtype]

                t = '{'
                for tx in range(nc):
                    t+= '*(' + symbol + '+' + \
                        self._components['LIB_PAIR_INDEX_0']
                    t+= '*' + str(nc) + '+' + str(tx) + '),'
                t = t[:-1] + '}'

                g = cgen.Value(dtype,isym+ncb)
                g = cgen.Initializer(g,t)

                kernel_gather.append(g)

        self._components['LIB_KERNEL_GATHER'] = kernel_gather

    def _generate_kernel_call(self):

        kernel_call = cgen.Module([cgen.Comment(
            '#### Kernel call arguments ####')])
        kernel_call_symbols = []
        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                kernel_call_symbols.append(dat[0])

        for i, dat in enumerate(self._dat_dict.items()):

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            if issubclass(type(obj), data.GlobalArrayClassic):
                kernel_call_symbols.append(symbol+'_c')
            elif issubclass(type(obj), host._Array):
                kernel_call_symbols.append(symbol)
            elif issubclass(type(obj), host.Matrix):
                call_symbol = symbol + '_c'
                kernel_call_symbols.append(call_symbol)

                nc = str(obj.ncomp)
                _ishift = '+' + self._components['LIB_PAIR_INDEX_0'] + '*' + nc
                _jshift = '+' + self._components['LIB_PAIR_INDEX_1'] + '*' + nc

                if mode.write and obj.ncomp <= self._gather_size_limit:
                    isym = '&'+ symbol+'i[0]'
                else:
                    isym = symbol + _ishift
                jsym = symbol + _jshift
                g = cgen.Value('_'+symbol+'_t', call_symbol)
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


    def _generate_lib_inner_loop(self):
        i = self._components['LIB_PAIR_INDEX_0']
        j = self._components['LIB_PAIR_INDEX_1']
        self._components['LIB_LOOP_J_PREPARE'] = cgen.Module([
            cgen.Line('const int _icell = _CRL['+i+'];'),
            cgen.Line('int * _JJSTORE = _JSTORE['+self._components[
                'OMP_THREAD_INDEX_SYM']+'];'),
            cgen.Line('int _nn = 0;'),
        ])

        b = self._components['LIB_INNER_LOOP_BLOCK']
        self._components['LIB_INNER_LOOP'] = cgen.Module([
                cgen.For('int _k=0', '_k<27', '_k++', b),
                cgen.For(
                    'int _k2=0','_k2<_nn','_k2++',
                    cgen.Block([
                        cgen.Line('const int '+j+' = _JJSTORE[_k2];' ),
                        self._components['LIB_KERNEL_CALL'],
                    ])
                )
            ])

    def _generate_lib_outer_loop(self):

        block = cgen.Block([self._components['LIB_KERNEL_GATHER'],
                            self._components['LIB_LOOP_J_PREPARE'],
                            self._components['LIB_INNER_LOOP'],
                            self._components['LIB_KERNEL_SCATTER']])

        i = self._components['LIB_PAIR_INDEX_0']

        shared = ''
        for sx in self._components['OMP_SHARED_SYMS']:
            shared+= sx+','
        shared = shared[:-1]
        pragma = cgen.Pragma('omp parallel for default(none) schedule(static) shared(' + shared + ')')
        if runtime.OMP_NUM_THREADS is None:
            pragma = cgen.Comment(pragma)

        loop = cgen.Module([
            cgen.Line('omp_set_num_threads(_NUM_THREADS);'),
            pragma,
            cgen.For('int ' + i + '=0',
                    i + '<_N_LOCAL',
                    i+'++',
                    block)
        ])

        self._components['LIB_OUTER_LOOP'] = loop


    def _init_jstore(self, cell2part):
        n = cell2part.max_cell_contents_count * 27
        if self._jstore[0].ncomp < n:
            self._jstore = [host.Array(ncomp=100+n, dtype=ctypes.c_int) for tx\
                            in range(runtime.NUM_THREADS)]

        return (ctypes.POINTER(ctypes.c_int) * runtime.NUM_THREADS)(*[
            self._jstore[tx].ctypes_data for tx in range(runtime.NUM_THREADS)
        ])

    def _get_class_lib_args(self, cell2part):
        assert ctypes.c_int == cell2part.cell_list.dtype
        assert ctypes.c_int == cell2part.cell_reverse_lookup.dtype
        assert ctypes.c_int == cell2part.cell_contents_count.dtype
        jstore = self._init_jstore(cell2part)
        offset = cell2part.cell_list.end - cell2part.domain.cell_count
        return [
            ctypes.c_int(runtime.NUM_THREADS),
            ctypes.c_int(cell2part.num_particles),
            ctypes.c_int(offset),
            cell2part.cell_list.ctypes_data,
            cell2part.cell_reverse_lookup.ctypes_data,
            cell2part.cell_contents_count.ctypes_data,
            self._offset_list.ctypes_data,
            ctypes.byref(jstore),
            self.loop_timer.get_python_parameters()
        ]

    def _get_dat_lib_args(self, dats):
        args = []
        for dat_orig in self._dat_dict.values(dats):
            assert type(dat_orig) is tuple
            obj = dat_orig[0]
            mode = dat_orig[1]
            if issubclass(type(obj), data.GlobalArrayClassic):
                args.append(
                    obj.ctypes_data_access(mode, pair=True, threaded=True)
                )
            else:
                args.append(obj.ctypes_data_access(mode, pair=True))

        return args

    def _post_execute_dats(self, dats):
        for dat_orig in self._dat_dict.values(dats):
            assert type(dat_orig) is tuple
            obj = dat_orig[0]
            mode = dat_orig[1]
            if issubclass(type(obj), data.GlobalArrayClassic):
                obj.ctypes_data_post(mode, threaded=True)
            else:
                obj.ctypes_data_post(mode)


    def execute(self, n=None, dat_dict=None, static_args=None):

        _group = self._group # could be None
        if _group is None:
            for pd in self._dat_dict.items(dat_dict):
                if issubclass(type(pd[1][0]), data.PositionDat):
                    _group = pd[1][0].group
                    break
            self._make_cell_list(_group)

        assert _group is not None, "no group"
        cell2part = _group.get_cell_to_particle_map()
        cell2part.check()
        self._make_cell_list(_group)

        args = []
        # Add static arguments to launch command
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            args += self._kernel.static_args.get_args(static_args)

        # Add pointer arguments to launch command
        self._init_dat_lib_args(dat_dict)
        args+=self._get_dat_lib_args(dat_dict)

        # Rebuild neighbour list potentially
        self._invocations += 1

        args2 = self._get_class_lib_args(cell2part)

        args = args2 + args

        # Execute the kernel over all particle pairs.
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        self._update_opt()
        self._post_execute_dats(dat_dict)






