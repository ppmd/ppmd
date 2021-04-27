from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import cgen

# package level
from ppmd import runtime, host, opt, data, access
from ppmd.loop.particle_loop import ParticleLoop
from ppmd.lib.common import ctypes_map, OMP_DECOMP_HEADER

import ppmd.sycl.sycl_runtime
from ppmd.sycl.sycl_build import sycl_simple_lib_creator, CC
from ppmd.modules.dsl_kernel_symbols import DSLKernelSymSub


def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)

class SYCLParticleLoopBasic:
 
    def __init__(self, kernel=None, dat_dict=None):

        self._dat_dict = access.DatArgStore(
            self._get_allowed_types(),
            dat_dict
        )

        self._kernel = kernel

        self.loop_timer = ppmd.modules.code_timer.LoopTimer()
        self.wrapper_timer = ppmd.opt.Timer(runtime.TIMER)

        self._components = None

        self._generate()
        
        #print(self._components['LIB_SRC'])
        self._lib = sycl_simple_lib_creator(self._generate_header_source(),
                                             self._components['LIB_SRC'],
                                             self._kernel.name)

        self._group = None

        for pd in self._dat_dict.items():

            if issubclass(type(pd[1][0]), data.ParticleDat):
                if pd[1][0].group is not None:
                    self._group = pd[1][0].group
                    break

    def _generate_kernel(self):
        k = DSLKernelSymSub(kernel=self._kernel.code)
        self._components['KERNEL'] = k  

    def _generate(self):
        self._init_components()
        self._generate_kernel()

        self._generate_kernel_call()
        self._generate_lib_specific_args()
        self._generate_kernel_arg_decls()
        self._generate_kernel_headers()


        self._generate_lib_outer_loop()
        self._generate_lib_func()
        self._generate_lib_src()


    def _get_allowed_types(self):
        return {
            data.ScalarArray: (access.READ, access._INTERNAL_RW),
            data.ParticleDat: access.all_access_types,
            data.PositionDat: access.all_access_types,
            data.GlobalArrayClassic: (access.INC_ZERO, access.INC, access.READ),
        }


    def _init_components(self):
        self._components = {
            'LIB_PAIR_INDEX_0': '_i',
            'LIB_NAME': str(self._kernel.name) + '_wrapper',
            'LIB_HEADERS': [cgen.Include('omp.h', system=True), cgen.Line(OMP_DECOMP_HEADER)],
            'OMP_THREAD_INDEX_SYM': '_threadid',
            "LOCAL_MEMORY_INIT": "",
            "LOCAL_MEMORY_FINALISE": "",
            "REDUCTION_STORE":"",
        }

    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
                cgen.Value('sycl::queue', '*_queue'),
            cgen.Const(cgen.Value(host.int32_str, '_N_LOCAL')),
            self.loop_timer.get_cpp_arguments_ast()
        ]

    def _generate_lib_src(self):
        self._components['LIB_SRC'] = cgen.Module([
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

        #define _WORKGROUPSIZE %(WORKGROUPSIZE)s

        extern "C" %(FUNC_DEC)s
        '''
        d = {
            'INCLUDED_HEADERS': str(self._components['KERNEL_HEADERS']),
            'FUNC_DEC': str(self._components['LIB_FUNC'].fdecl),
            'LIB_DIR': runtime.LIB_DIR,
            'WORKGROUPSIZE': ppmd.sycl.sycl_runtime.WORK_GROUP_SIZE,
        }
        return code % d


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


    def _generate_kernel_arg_decls(self):
        
        _kernel_lib_arg_decls = []

        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                arg = cgen.Const(cgen.Value(host.ctypes_map[dat[1]], dat[0]))
                _kernel_lib_arg_decls.append(arg)

        for i, dat in enumerate(self._dat_dict.items()):

            assert type(dat[1]) is tuple, "Access descriptors not found"

            obj = dat[1][0]
            mode = dat[1][1]
            sym = dat[0]
            dtype = dat[1][0].dtype

            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype],
                                          Restrict(CC.restrict_keyword, sym))
                                      )

            if issubclass(type(obj), data.GlobalArrayClassic):
                pass

            if issubclass(type(obj), host._Array):
                kernel_arg = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype],
                                              Restrict(CC.restrict_keyword, sym))
                                          )
                if not mode.write:
                    kernel_arg = cgen.Const(kernel_arg)


                if mode.write is True:
                    assert issubclass(type(obj), data.GlobalArrayClassic) or mode == access._INTERNAL_RW, \
                        "global array must be a thread safe type for write access. Type is:" + str(type(obj))


            elif issubclass(type(obj), host.Matrix):
                # MAKE STRUCT TYPE
                ti = cgen.Pointer(cgen.Value(ctypes_map(dtype),
                                             Restrict(CC.restrict_keyword,'i')))
                if not mode.write:
                    ti = cgen.Const(ti)
                typename = f"_{sym}_t"


            if not mode.write:
                kernel_lib_arg = cgen.Const(kernel_lib_arg)

            _kernel_lib_arg_decls.append(kernel_lib_arg)

        self._components['KERNEL_LIB_ARG_DECLS'] = _kernel_lib_arg_decls


    def _generate_kernel_call(self):

        kernel = self._components["KERNEL"]
        kernel_call = cgen.Module([
            cgen.Comment('#### Kernel call arguments ####'),
        ])
        kernel_call_symbols = []
        sycl_accessors = []
        sycl_buffers = []

        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                kernel_call_symbols.append(dat[0])

        local_init = []
        local_finalise = []
        reduction_store = []

        _i = self._components['LIB_PAIR_INDEX_0']
        _i_local = f"{_i}_local"
        

        for i, dat in enumerate(self._dat_dict.items()):
            mode = dat[1][1]
            obj = dat[1][0]
            dtype = dat[1][0].dtype
            ctype = ctypes_map(dtype)
            nc = str(dat[1][0].ncomp)
            sym = dat[0]
            if issubclass(type(obj), host._Array):
                #if issubclass(type(dat[1][0]), data.GlobalArrayClassic):
                #    sym += '[' + self._components['OMP_THREAD_INDEX_SYM'] + ']'

                sycl_access_sym = '_sycl_access_' + sym
                sycl_buffer_sym = '_{}_sycl_buffer'.format(sym)
 
                sycl_buffer_creator = 'sycl::buffer<{}, 1>{}({}, sycl::range<1>({}));'.format(
                    ctype,
                    sycl_buffer_sym,
                    sym,
                    nc
                )

                if mode.write: # GA GlobalArray case
                    sycl_access_creator = f"""
                    sycl::accessor <{ctype}, 1, sycl::access::mode::read_write, sycl::access::target::local> 
                        {sycl_access_sym}(sycl::range<1>(_WORKGROUPSIZE * {nc}), _cgh);
                    """
                    
                    sycl_global_access_sym = f"{sycl_access_sym}_global"

                    sycl_local_access_sym = f"{sycl_access_sym}_local_mem"

                    local_init.append(
                        f"""
                            for(int _cx=0 ; _cx<{nc} ; _cx++) {{ {sycl_access_sym}[{_i_local} * {nc} + _cx] = 0; }}
                            auto {sycl_local_access_sym} = &{sycl_access_sym}[{_i_local}*{nc}];
                        """
                    )

                    local_finalise.append(
                        f"""
                        if ({_i_local} == 0){{
                            for(int _cx=0 ; _cx<{nc} ; _cx++) {{
                                {ctype} _red_tmp = 0;
                                for(int _lind=0 ; _lind<_item.get_local_range(0) ; _lind++) {{
                                    _red_tmp += {sycl_access_sym}[_lind * {nc} + _cx];
                                }}
                                {sycl_global_access_sym}[_item.get_group_linear_id() * {nc} + _cx] += _red_tmp;
                            }}
                        }}
                        """
                    )
                    sycl_buffer_creator += f"\nauto {sycl_global_access_sym} = static_cast<{ctype} *>(sycl::malloc_shared(sizeof({ctype}) * _WORKGROUPCOUNT * {nc}, *_queue));"

                    sycl_global_access = f"""
                    for (int _lind=0 ; _lind<(_WORKGROUPCOUNT * {nc}) ; _lind++){{
                        {sycl_global_access_sym}[_lind] = 0;
                    }}
                    """

                    sycl_access_creator += "\n" + sycl_global_access

                    reduction_store.append(
                        f"""

                        for(int _cx=0 ; _cx<{nc} ; _cx++){{
                            for(int _wx=0 ; _wx<_WORKGROUPCOUNT ; _wx++){{
                                {sym}[_cx] += {sycl_global_access_sym}[_wx * {nc} + _cx];
                            }}
                        }}
                        sycl::free({sycl_global_access_sym}, *_queue);
                        """
                    )

                    kernel.sub_sym(sym, sycl_local_access_sym)
                else:

                    sycl_access_creator = 'auto {0} = {1}.get_access<sycl::access::mode::{2}>(_cgh);'.format(
                        sycl_access_sym,
                        sycl_buffer_sym,
                        'read' if not mode.write else 'write'
                    )
                    kernel.sub_sym(sym, sycl_access_sym)


            elif issubclass(type(obj), host.Matrix):

                _ishift = _i + '*' + nc
                
                sycl_access_sym = '_sycl_access_' + sym
                sycl_buffer_sym = '_{}_sycl_buffer'.format(sym)

                sycl_buffer_creator = 'sycl::buffer<{}, 1>{}({}, sycl::range<1>(_N_LOCAL * {}));'.format(
                    ctype,
                    sycl_buffer_sym,
                    dat[0],
                    nc
                )

                sycl_access_creator = 'auto {0} = {1}.get_access<sycl::access::mode::{2}>(_cgh);'.format(
                    sycl_access_sym,
                    sycl_buffer_sym,
                    'read' if not mode.write else 'write'
                )

                inner_sym = '_inner_{}_i'.format(sym)
                kernel.sub_sym(sym + '.i', inner_sym)
                kernel_call.append(
                    cgen.Line(
                        "auto {} = &{}[{}];".format(
                            inner_sym,
                            sycl_access_sym,
                            _ishift
                        )
                    )
                )

            else:
                print("ERROR: Type not known")

            sycl_accessors.append(sycl_access_creator)
            sycl_buffers.append(sycl_buffer_creator)

        kernel_call.append(cgen.Comment('#### Kernel call ####'))

        kernel_call.append(cgen.Line(
            self._components['KERNEL'].kernel
        ))

        self._components['LIB_KERNEL_CALL'] = kernel_call
        self._components['SYCL_ACCESSORS'] = sycl_accessors
        self._components['SYCL_BUFFERS'] = sycl_buffers
        self._components['REDUCTION_STORE'] = '\n'.join(reduction_store)

        if len(local_init) > 0:
            self._components["LOCAL_MEMORY_INIT"] = "\n".join(local_init)
        if len(local_finalise) > 0:
            self._components["LOCAL_MEMORY_FINALISE"] = "_item.barrier(sycl::access::fence_space::local_space);\n"
            self._components["LOCAL_MEMORY_FINALISE"] += "\n".join(local_finalise)


    def _generate_lib_outer_loop(self):

        
        parallel_region = cgen.Block(
            (
            ## buffer creationi
            cgen.Line("""
            {{

                const int64_t _WORKGROUPCOUNT = (_N_LOCAL / _WORKGROUPSIZE) + 1;

                {BUFFERS}
                (*_queue).submit([&] (sycl::handler& _cgh) {{
                    {ACCESS}
                    _cgh.parallel_for<class _foo>(
                        sycl::nd_range<1>(_N_LOCAL, _WORKGROUPSIZE),
                        [=] (sycl::nd_item<1> _item) {{
                            auto {INDEX} = _item.get_global_linear_id();
                            auto {INDEX}_local = _item.get_local_linear_id();
                            
                            {LOCAL_MEMORY_INIT}

                            {KERNEL_CALL}

                            {LOCAL_MEMORY_FINALISE}
                        }}
                    );
                }});
                (*_queue).wait_and_throw();

                {REDUCTION_STORE}
            }}
                """.format(
                    BUFFERS='\n'.join(self._components['SYCL_BUFFERS']),
                    ACCESS='\n'.join(self._components['SYCL_ACCESSORS']),
                    INDEX=self._components['LIB_PAIR_INDEX_0'],
                    KERNEL_CALL=self._components['LIB_KERNEL_CALL'],
                    LOCAL_MEMORY_INIT=self._components["LOCAL_MEMORY_INIT"],
                    LOCAL_MEMORY_FINALISE=self._components["LOCAL_MEMORY_FINALISE"],
                    REDUCTION_STORE=self._components["REDUCTION_STORE"],
                )
                ),
            )
        )

        loop = cgen.Module([
            parallel_region,
        ])

        self._components['LIB_OUTER_LOOP'] = loop



    def execute(self, n=None, dat_dict=None, static_args=None):
        """
        C version of the pair_locate: Loop over all cells update forces and
         potential engery.
        """

        _N_LOCAL = None
        args = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            args += self._kernel.static_args.get_args(static_args)

        for dat in self._dat_dict.items(new_dats=dat_dict):
            obj = dat[1][0]
            mode = dat[1][1]

            if issubclass(type(obj), data.GlobalArrayClassic):
                args.append(obj.ctypes_data_access(mode, pair=False, threaded=False))
            else:
                args.append(obj.ctypes_data_access(mode, pair=False))

            if issubclass(type(obj), data.ParticleDat):
                _N_LOCAL = obj.npart_local


        '''Create arg list'''
        if n is None and _N_LOCAL is None:
            assert self._group is not None, "cannot determine number of particles"
            _N_LOCAL = ctypes.c_int(self._group.npart_local)

        if n is not None:
            _N_LOCAL = ctypes.c_int(n)

        assert _N_LOCAL is not None
        args2 = [ppmd.sycl.sycl_runtime.queue.queue, _N_LOCAL]

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
            if issubclass(type(obj), data.GlobalArrayClassic):
                obj.ctypes_data_post(mode, threaded=True)
            else:
                obj.ctypes_data_post(mode)

