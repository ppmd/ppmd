from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


# system level
import ctypes
import cgen
import os

# package level
from ppmd import build
from ppmd import runtime
from ppmd import host
from ppmd import opt
from ppmd import data

from particle_loop import ParticleLoop

def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)


class ParticleLoopOMP(ParticleLoop):

    def _init_components(self):
        self._components = {
            'LIB_PAIR_INDEX_0': '_i',
            'LIB_NAME': str(self._kernel.name) + '_wrapper',
            'LIB_HEADERS': [cgen.Include('omp.h', system=True),],
            'OMP_THREAD_INDEX_SYM': '_threadid',
            'OMP_SHARED_SYMS': []
            }

    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, '_NUM_THREADS')),
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

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype],
                                          Restrict(self._cc.restrict_keyword, symbol))
                                      )

            if type(obj) is data.GlobalArrayThreaded:
                kernel_lib_arg = cgen.Pointer(kernel_lib_arg)

            if issubclass(type(obj), host._Array):
                kernel_arg = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype],
                                              Restrict(self._cc.restrict_keyword, symbol))
                                          )
                if not mode.write:
                    kernel_arg = cgen.Const(kernel_arg)

                _kernel_arg_decls.append(kernel_arg)

                if mode.write is True:
                    assert type(obj) is data.GlobalArrayThreaded, "global array must be a thread safe type for \
                    write access. Type is:" + str(type(obj))


            elif issubclass(type(dat[1][0]), host.Matrix):
                # MAKE STRUCT TYPE
                dtype = dat[1][0].dtype
                ti = cgen.Pointer(cgen.Value(cgen.dtype_to_ctype(dtype),
                                             Restrict(self._cc.restrict_keyword,'i')))
                if not dat[1][1].write:
                    ti = cgen.Const(ti)
                typename = '_'+dat[0]+'_t'
                _kernel_structs.append(cgen.Typedef(cgen.Struct('', [ti], typename)))

                # MAKE STRUCT ARG
                _kernel_arg_decls.append(cgen.Value(typename, dat[0]))


            if not dat[1][1].write:
                kernel_lib_arg = cgen.Const(kernel_lib_arg)

            _kernel_lib_arg_decls.append(kernel_lib_arg)

        self._components['KERNEL_ARG_DECLS'] = _kernel_arg_decls
        self._components['KERNEL_LIB_ARG_DECLS'] = _kernel_lib_arg_decls
        self._components['KERNEL_STRUCT_TYPEDEFS'] = _kernel_structs


    def _generate_kernel_call(self):

        kernel_call = cgen.Module([
            cgen.Comment('#### Kernel call arguments ####'),
            cgen.Initializer(cgen.Const(cgen.Value(
                'int', self._components['OMP_THREAD_INDEX_SYM'])),
                'omp_get_thread_num()')
        ])
        kernel_call_symbols = []
        shared_syms = self._components['OMP_SHARED_SYMS']

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
                sym = dat[0]
                if type(dat[1][0]) is data.GlobalArrayThreaded:
                    sym += '[' + self._components['OMP_THREAD_INDEX_SYM'] + ']'
                kernel_call_symbols.append(sym)
                shared_syms.append(dat[0])

            elif issubclass(type(dat[1][0]), host.Matrix):
                call_symbol = dat[0] + '_c'
                kernel_call_symbols.append(call_symbol)

                nc = str(dat[1][0].ncomp)
                _ishift = '+' + self._components['LIB_PAIR_INDEX_0'] + '*' + nc

                isym = dat[0] + _ishift
                g = cgen.Value('_'+dat[0]+'_t', call_symbol)
                g = cgen.Initializer(g, '{ ' + isym + '}')

                kernel_call.append(g)
                shared_syms.append(dat[0])

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

        block = cgen.Block([self._components['LIB_KERNEL_CALL']])

        i = self._components['LIB_PAIR_INDEX_0']

        shared = ''
        for sx in self._components['OMP_SHARED_SYMS']:
            shared+= sx+','
        shared = shared[:-1]

        loop = cgen.Module([
            cgen.Line('omp_set_num_threads(_NUM_THREADS);'),
            cgen.Pragma('omp parallel for default(none) shared(' + shared + ')'),
            cgen.For('int ' + i + '=0',
                    i + '<_N_LOCAL',
                    i+'++',
                    block)
        ])

        self._components['LIB_OUTER_LOOP'] = loop



    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and
         potential engery.
        """

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
                dat_orig[0].ctypes_data_access(dat_orig[1], pair=False)


        '''Add pointer arguments to launch command'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:

                if type(dat_orig[0]) is data.GlobalArrayThreaded:
                    args.append(dat_orig[0].ctypes_data_access(dat_orig[1], pair=False, threaded=True))
                else:
                    args.append(dat_orig[0].ctypes_data_access(dat_orig[1], pair=False))


                if issubclass(type(dat_orig[0]), data.ParticleDat):
                    _N_LOCAL = dat_orig[0].npart_local
            else:
                raise RuntimeError
                #args.append(dat_orig.ctypes_data)


        '''Create arg list'''
        if n is None:
            if self._group is not None:
                _N_LOCAL = ctypes.c_int(self._group.npart_local)

        else:
            _N_LOCAL = ctypes.c_int(n)

        args2 = [ctypes.c_int(runtime.NUM_THREADS), _N_LOCAL]

        args2.append(self.loop_timer.get_python_parameters())

        args = args2 + args

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':execute_internal'
        ] = (self.loop_timer.time)

        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                if type(dat_orig[0]) is data.GlobalArrayThreaded:
                    dat_orig[0].ctypes_data_post(dat_orig[1], threaded=True)
                else:
                    dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()

