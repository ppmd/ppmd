# system level
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import cgen
import ctypes

# package level

from ppmd import data, runtime, host, access

from neighbourlist import PairLoopNeighbourListNS, Restrict

class PairLoopNeighbourListNSOMP(PairLoopNeighbourListNS):

    @staticmethod
    def _get_allowed_types():
        return {
            data.ScalarArray: (access.READ,),
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
             'LIB_HEADERS': [cgen.Include('omp.h', system=True),],
             'OMP_THREAD_INDEX_SYM': '_threadid',
             'OMP_SHARED_SYMS': ['_START_POINTS', '_NLIST']
         }

    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, '_NUM_THREADS')),
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
            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype],
                                          Restrict(self._cc.restrict_keyword, symbol))
                                      )

            if issubclass(type(obj), data.GlobalArrayClassic):
                kernel_lib_arg = cgen.Pointer(kernel_lib_arg)

            if issubclass(type(obj), host._Array):
                kernel_arg = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype],
                                              Restrict(self._cc.restrict_keyword, symbol))
                                          )
                if not mode.write:
                    kernel_arg = cgen.Const(kernel_arg)
                _kernel_arg_decls.append(kernel_arg)

                if mode.write is True:
                    assert issubclass(type(obj), data.GlobalArrayClassic), "global array must be a thread safe type for \
                    write access. Type is:" + str(type(obj))


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
                    t+= '*(' + symbol + '+' + self._components['LIB_PAIR_INDEX_0']
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


    def _generate_lib_outer_loop(self):

        block = cgen.Block([self._components['LIB_KERNEL_GATHER'],
                            self._components['LIB_INNER_LOOP'],
                            self._components['LIB_KERNEL_SCATTER']])

        i = self._components['LIB_PAIR_INDEX_0']

        shared = ''
        for sx in self._components['OMP_SHARED_SYMS']:
            shared+= sx+','
        shared = shared[:-1]
        pragma = cgen.Pragma('omp parallel for default(none) shared(' + shared + ')')
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


    def _get_class_lib_args(self):

        neighbour_list = PairLoopNeighbourListNS._neighbour_list_dict_PNLNS[self._key]
        _N_LOCAL = ctypes.c_int(neighbour_list.n_local)
        _STARTS = neighbour_list.neighbour_starting_points.ctypes_data
        _LIST = neighbour_list.list.ctypes_data

        return [
            ctypes.c_int(runtime.NUM_THREADS),
            _N_LOCAL,
            _STARTS,
            _LIST,
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








