from __future__ import print_function, division, absolute_import

import ppmd.modules.code_timer
import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


# system level
import ctypes

import cgen
import os

# package level
from ppmd import runtime, host, opt, data, access
from ppmd.lib import build
from ppmd.lib.common import ctypes_map

def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)


class ParticleLoop(object):


    def __init__(self, kernel=None, dat_dict=None):

        self._dat_dict = access.DatArgStore(
            self._get_allowed_types(),
            dat_dict
        )

        self._cc = build.TMPCC

        self._temp_dir = runtime.BUILD_DIR
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        self._kernel = kernel

        self.loop_timer = ppmd.modules.code_timer.LoopTimer()
        self.wrapper_timer = ppmd.opt.Timer(runtime.TIMER)

        self._components = None

        self._generate()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._components['LIB_SRC'],
                                             self._kernel.name,
                                             CC=self._cc)

        self._group = None

        for pd in self._dat_dict.items():

            if issubclass(type(pd[1][0]), data.ParticleDat):
                if pd[1][0].group is not None:
                    self._group = pd[1][0].group
                    break

        # assert self._group is not None, "No cell to particle map found"

    def _get_allowed_types(self):
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

        self._generate_kernel_call()

        self._generate_lib_outer_loop()
        self._generate_lib_func()
        self._generate_lib_src()


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


    def _generate_map_macros(self):

        g = cgen.Module([cgen.Comment('#### KERNEL_MAP_MACROS ####')])

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
                g.append(cgen.Define(dat[0]+'(x)', '('+dat[0]+'[(x)])'))
            if issubclass(type(dat[1][0]), host.Matrix):
                g.append(cgen.Define(dat[0]+'(y)', dat[0]+'.i[(y)]'))

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


    def _generate_kernel_call(self):

        kernel_call = cgen.Module([cgen.Comment('#### Kernel call arguments ####')])
        kernel_call_symbols = []
        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                kernel_call_symbols.append(dat[0])

        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), host._Array):
                kernel_call_symbols.append(dat[0])
            elif issubclass(type(dat[1][0]), host.Matrix):
                call_symbol = dat[0] + '_c'
                kernel_call_symbols.append(call_symbol)

                nc = str(dat[1][0].ncomp)
                _ishift = '+' + self._components['LIB_PAIR_INDEX_0'] + '*' + nc

                isym = dat[0] + _ishift
                g = cgen.Value('_'+dat[0]+'_t', call_symbol)
                g = cgen.Initializer(g, '{ ' + isym + '}')

                kernel_call.append(g)

            else:
                raise RuntimeError("ERROR: Type not known")

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

        args = []
        # Add static arguments to launch command
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            args += self._kernel.static_args.get_args(static_args)

        # pointer args
        for dat in self._dat_dict.items(new_dats=dat_dict):
            obj = dat[1][0]
            mode = dat[1][1]
            args.append(obj.ctypes_data_access(mode, pair=False))
            if issubclass(type(obj), data.ParticleDat):
                _N_LOCAL = obj.npart_local

        # get a number of particles
        if n is None:
            if self._group is not None:
                _N_LOCAL = ctypes.c_int(self._group.npart_local)

        else:
            _N_LOCAL = ctypes.c_int(n)

        args2 = [_N_LOCAL]

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

        for dat in self._dat_dict.items(new_dats=dat_dict):
            obj = dat[1][0]
            mode = dat[1][1]
            obj.ctypes_data_post(mode)
